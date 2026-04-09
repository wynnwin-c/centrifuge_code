import torch
import torch.nn as nn

from .cnn_1d import cnn_features
from .classifier import Classifier
from .transfer_losses import TransferLoss


class SpeedFiLM(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def forward(self, feat: torch.Tensor, speed: torch.Tensor | None):
        if speed is None:
            return feat
        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        gamma_beta = self.mlp(speed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 1.0 + gamma
        return gamma * feat + beta


class TransferNet(nn.Module):
    def __init__(self, num_class, model_name, in_channel, transfer_loss, max_iter, trade_off_adversarial, lam_adversarial, batch_size, shared_classes=None, aux_source_weight=0.0, aux_target_weight=0.0, pseudo_threshold=0.0, pseudo_topk=0, pseudo_warmup_epochs=0):
        super().__init__()
        self.num_class = num_class
        self.transfer_loss = transfer_loss
        self.batch_size = batch_size
        self.shared_classes = list(shared_classes or [])
        self.aux_source_weight = float(aux_source_weight)
        self.aux_target_weight = float(aux_target_weight)
        self.pseudo_threshold = float(pseudo_threshold)
        self.pseudo_topk = int(pseudo_topk)
        self.pseudo_warmup_epochs = int(pseudo_warmup_epochs)
        if model_name != 'cnn_features':
            raise ValueError(f'Unsupported model_name: {model_name}')
        self.backbone = cnn_features(in_channel=in_channel)
        self.film = SpeedFiLM(self.backbone.output_num())
        self.classifier = Classifier(
            input_num=self.backbone.output_num(),
            hidden_num1=self.backbone.hidden_num1(),
            hidden_num2=self.backbone.hidden_num2(),
            output_num=self.num_class,
        )
        self.aux_classifier = Classifier(
            input_num=self.backbone.output_num(),
            hidden_num1=self.backbone.hidden_num1(),
            hidden_num2=self.backbone.hidden_num2(),
            output_num=2,
        )
        self.transfer_loss_args = {
            'loss_type': self.transfer_loss,
            'num_class': num_class,
            'max_iter': max_iter,
            'input_num': self.backbone.output_num(),
            'hidden_num1': self.backbone.adv_hidden_num1(),
            'hidden_num2': self.backbone.adv_hidden_num2(),
            'trade_off_adversarial': trade_off_adversarial,
            'lam_adversarial': lam_adversarial,
        }
        if self.transfer_loss in {'bsan', 'bsdan'}:
            self.transfer_loss_args['shared_classes'] = self.shared_classes
        self.adapt_loss = TransferLoss(**self.transfer_loss_args)
        self.criterion = nn.CrossEntropyLoss()

    def encode(self, x, speed=None):
        feat = self.backbone(x)
        return feat

    def forward(self, source, target, source_label, source_speed=None, target_speed=None, epoch=1, epochs=1):
        len_share = source.size(0) - target.size(0)
        if len_share > 0:
            aug_source = source
            aug_source_label = source_label
            aug_source_speed = source_speed
            source_label = aug_source_label[:target.size(0)]
            source = aug_source[:target.size(0)]
            source_speed = aug_source_speed[:target.size(0)] if aug_source_speed is not None else None
            middle = aug_source[target.size(0):]
            middle_speed = aug_source_speed[target.size(0):] if aug_source_speed is not None else None
        else:
            middle = None
            middle_speed = None

        source = self.encode(source, source_speed)
        target = self.encode(target, target_speed)
        if middle is not None:
            middle = self.encode(middle, middle_speed)

        source_cls = self.classifier(source)
        cls_loss = self.criterion(source_cls, source_label)
        target_cls = self.classifier(target)

        aux_loss = torch.tensor(0.0, device=source.device)
        if self.aux_source_weight > 0.0:
            source_aux_logits = self.aux_classifier(source)
            source_aux_labels = (source_label != 0).long()
            aux_loss = aux_loss + self.aux_source_weight * self.criterion(source_aux_logits, source_aux_labels)

        enable_target_pseudo = self.aux_target_weight > 0.0 and self.shared_classes and epoch > self.pseudo_warmup_epochs
        if enable_target_pseudo:
            shared_target_probs = torch.softmax(target_cls[:, self.shared_classes], dim=1)
            selected_indices = []
            selected_labels = []
            for local_cls, global_cls in enumerate(self.shared_classes):
                cls_scores = shared_target_probs[:, local_cls]
                if self.pseudo_topk > 0:
                    k = min(self.pseudo_topk, cls_scores.numel())
                    topk_idx = torch.topk(cls_scores, k=k).indices
                    selected_indices.append(topk_idx)
                    selected_labels.append(torch.full((k,), 0 if global_cls == 0 else 1, device=cls_scores.device, dtype=torch.long))
                else:
                    mask = cls_scores >= self.pseudo_threshold
                    idx = torch.nonzero(mask, as_tuple=False).view(-1)
                    if idx.numel() > 0:
                        selected_indices.append(idx)
                        selected_labels.append(torch.full((idx.numel(),), 0 if global_cls == 0 else 1, device=cls_scores.device, dtype=torch.long))
            if selected_indices:
                idx_all = torch.cat(selected_indices, dim=0)
                y_all = torch.cat(selected_labels, dim=0)
                target_aux_logits = self.aux_classifier(target[idx_all])
                aux_loss = aux_loss + self.aux_target_weight * self.criterion(target_aux_logits, y_all)

        kwargs = {}
        if self.transfer_loss in {'bsan', 'bsdan'}:
            kwargs['len_share'] = len_share
            target_for_transfer = target
            if middle is not None:
                target_for_transfer = torch.cat((target, middle), dim=0)
            kwargs['source_logits'] = torch.softmax(source_cls, dim=1)
            kwargs['target_logits'] = torch.softmax(self.classifier(target_for_transfer), dim=1)
        elif self.transfer_loss in {'ladv', 'sladv', 'san', 'edann'}:
            target_for_transfer = target
            kwargs['source_logits'] = torch.softmax(source_cls, dim=1)
            kwargs['target_logits'] = torch.softmax(target_cls, dim=1)
        else:
            target_for_transfer = target

        transfer_loss = self.adapt_loss(source, target_for_transfer, **kwargs)
        return cls_loss + aux_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.backbone.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.film.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.aux_classifier.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.transfer_loss in {'ladv', 'sladv', 'san', 'bsan', 'bsdan'}:
            params.append({'params': self.adapt_loss.loss_func.local_classifier.parameters(), 'lr': 1.0 * initial_lr})
        elif self.transfer_loss in {'adv', 'edann'}:
            params.append({'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr})
        return params

    def predict(self, x, speed=None):
        return self.classifier(self.encode(x, speed))

    def infer(self, x, speed=None):
        return self.predict(x, speed)

    def predict_binary(self, x, speed=None):
        return self.aux_classifier(self.encode(x, speed))

    def infer_binary(self, x, speed=None):
        return self.predict_binary(x, speed)

    def epoch_based_processing(self, *args, **kwargs):
        return None
