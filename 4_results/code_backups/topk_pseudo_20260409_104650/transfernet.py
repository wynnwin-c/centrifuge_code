import torch
import torch.nn as nn

from .cnn_1d import cnn_features
from .classifier import Classifier
from .transfer_losses import TransferLoss


class TransferNet(nn.Module):
    def __init__(self, num_class, model_name, in_channel, transfer_loss, max_iter, trade_off_adversarial, lam_adversarial, batch_size, shared_classes=None, aux_source_weight=0.0, aux_target_weight=0.0, pseudo_threshold=0.0):
        super().__init__()
        self.num_class = num_class
        self.transfer_loss = transfer_loss
        self.batch_size = batch_size
        self.shared_classes = list(shared_classes or [])
        self.aux_source_weight = float(aux_source_weight)
        self.aux_target_weight = float(aux_target_weight)
        self.pseudo_threshold = float(pseudo_threshold)
        if model_name != 'cnn_features':
            raise ValueError(f'Unsupported model_name: {model_name}')
        self.backbone = cnn_features(in_channel=in_channel)
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
        if self.transfer_loss == 'bsan':
            self.transfer_loss_args['shared_classes'] = self.shared_classes
        self.adapt_loss = TransferLoss(**self.transfer_loss_args)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        len_share = source.size(0) - target.size(0)
        if len_share > 0:
            aug_source = source
            aug_source_label = source_label
            source_label = aug_source_label[:target.size(0)]
            source = aug_source[:target.size(0)]
            middle = aug_source[target.size(0):]
        else:
            middle = None

        source = self.backbone(source)
        target = self.backbone(target)
        if middle is not None:
            middle = self.backbone(middle)

        source_cls = self.classifier(source)
        target_cls = self.classifier(target)
        cls_loss = self.criterion(source_cls, source_label)

        aux_loss = torch.tensor(0.0, device=source.device)
        source_aux_logits = self.aux_classifier(source)
        source_aux_labels = (source_label != 0).long()
        if self.aux_source_weight > 0.0:
            aux_loss = aux_loss + self.aux_source_weight * self.criterion(source_aux_logits, source_aux_labels)

        if self.aux_target_weight > 0.0 and self.shared_classes:
            shared_target_probs = torch.softmax(target_cls[:, self.shared_classes], dim=1)
            conf, idx = shared_target_probs.max(dim=1)
            mask = conf >= self.pseudo_threshold
            if mask.any():
                global_pred = torch.tensor(self.shared_classes, device=idx.device, dtype=torch.long)[idx[mask]]
                pseudo_aux_labels = (global_pred != 0).long()
                target_aux_logits = self.aux_classifier(target[mask])
                aux_loss = aux_loss + self.aux_target_weight * self.criterion(target_aux_logits, pseudo_aux_labels)

        kwargs = {
            'source_logits': torch.softmax(source_cls, dim=1),
            'target_logits': torch.softmax(target_cls, dim=1),
        }
        target_for_transfer = target
        if self.transfer_loss == 'bsan':
            if middle is not None:
                target_for_transfer = torch.cat((target, middle), dim=0)
                kwargs['target_logits'] = torch.softmax(self.classifier(target_for_transfer), dim=1)
            kwargs['len_share'] = len_share
        transfer_loss = self.adapt_loss(source, target_for_transfer, **kwargs)
        return cls_loss + aux_loss, transfer_loss

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.backbone.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.aux_classifier.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.transfer_loss in {'ladv', 'sladv', 'san', 'bsan'}:
            params.append({'params': self.adapt_loss.loss_func.local_classifier.parameters(), 'lr': 1.0 * initial_lr})
        elif self.transfer_loss == 'adv':
            params.append({'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr})
        return params

    def predict(self, x):
        return self.classifier(self.backbone(x))

    def infer(self, x):
        return self.predict(x)

    def epoch_based_processing(self, *args, **kwargs):
        return None
