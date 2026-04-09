import numpy as np
import torch
import torch.nn as nn


def calc_coeff(iter_num, high, low, alpha, max_iter):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class AdversarialNet(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter=None):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_num, hidden_num1), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(hidden_num1, hidden_num2), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(hidden_num2, 1)
        self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.high = 1.0
        self.low = 0.0
        self.alpha = 10
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = self.lam_adversarial if self.trade_off_adversarial == 'Cons' else calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class AdversarialLoss(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter):
        super().__init__()
        self.domain_classifier = AdversarialNet(input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter)

    def get_adversarial_result(self, x, source=True):
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        domain_label = torch.ones(len(x), 1, device=device) if source else torch.zeros(len(x), 1, device=device)
        return nn.BCELoss()(domain_pred, domain_label)

    def forward(self, source, target):
        return 0.5 * (self.get_adversarial_result(source, True) + self.get_adversarial_result(target, False))


class LADVLoss(AdversarialLoss):
    def __init__(self, num_class, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter):
        super().__init__(input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter)
        self.num_class = num_class
        self.local_classifier = nn.ModuleList([AdversarialNet(input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter) for _ in range(num_class)])

    def get_local_adversarial_result(self, x, logits, source=True):
        loss_fn = nn.BCELoss()
        loss_adv = torch.tensor(0.0, device=x.device)
        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0], 1))
            features_c = logits_c * x
            domain_pred = self.local_classifier[c](features_c)
            domain_label = torch.ones(len(x), 1, device=x.device) if source else torch.zeros(len(x), 1, device=x.device)
            loss_adv = loss_adv + loss_fn(domain_pred, domain_label)
        return loss_adv

    def forward(self, source, target, source_logits, target_logits):
        return 0.5 * (self.get_local_adversarial_result(source, source_logits, True) + self.get_local_adversarial_result(target, target_logits, False))


class SLADVLoss(LADVLoss):
    def get_local_adversarial_result(self, x, source_logits, target_logits, source=True):
        loss_fn = nn.BCELoss()
        loss_adv = torch.tensor(0.0, device=x.device)
        class_weight = torch.mean(target_logits, 0)
        class_weight = (class_weight / torch.max(class_weight).clamp_min(1e-8)).to(x.device).view(-1)
        logits = source_logits if source else target_logits
        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0], 1))
            features_c = logits_c * x
            self.local_classifier[c].high = float(class_weight[c].item())
            domain_pred = self.local_classifier[c](features_c)
            domain_label = torch.ones(len(x), 1, device=x.device) if source else torch.zeros(len(x), 1, device=x.device)
            loss_adv = loss_adv + loss_fn(domain_pred, domain_label)
        return loss_adv

    def forward(self, source, target, source_logits, target_logits):
        return 0.5 * (self.get_local_adversarial_result(source, source_logits, target_logits, True) + self.get_local_adversarial_result(target, source_logits, target_logits, False))


class SANLoss(SLADVLoss):
    def entropy_loss(self, input_):
        mask = input_.ge(0.000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def forward(self, source, target, source_logits, target_logits):
        local_loss = super().forward(source, target, source_logits, target_logits)
        return local_loss + 0.1 * self.entropy_loss(target_logits)


class BSANLoss(SLADVLoss):
    def entropy_loss(self, input_):
        mask = input_.ge(0.000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def forward(self, source, target, source_logits, target_logits, len_share):
        local_loss = super().forward(source, target, source_logits, target_logits)
        if len_share > 0:
            target_logits = target_logits[:source_logits.size(0)]
        return local_loss + 0.1 * self.entropy_loss(target_logits)


class BSDANLoss(LADVLoss):
    def entropy_loss(self, input_):
        mask = input_.ge(0.000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def forward(self, source, target, source_logits, target_logits, len_share, rho=0.0):
        loss_fn = nn.BCELoss()
        src_probs = source_logits.detach()
        tgt_probs = target_logits.detach()
        class_weight = torch.mean(tgt_probs, 0).to(source.device).view(-1)
        total_loss = torch.tensor(0.0, device=source.device)
        for c in range(self.num_class):
            src_prob = src_probs[:, c].reshape((src_probs.shape[0], 1))
            tgt_prob = tgt_probs[:, c].reshape((tgt_probs.shape[0], 1))
            feat_s_c = src_prob * source
            feat_t_c = tgt_prob * target
            pred_s = self.local_classifier[c](feat_s_c)
            pred_t = self.local_classifier[c](feat_t_c)
            label_s = torch.ones_like(pred_s)
            label_t = torch.zeros_like(pred_t)
            loss_s = loss_fn(pred_s, label_s)
            loss_t = loss_fn(pred_t, label_t)
            loss_exp = loss_fn(pred_s, torch.zeros_like(pred_s))
            total_loss = total_loss + class_weight[c] * (loss_s + loss_t + rho * loss_exp)
        return total_loss + 0.1 * self.entropy_loss(tgt_probs)


class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'bsan':
            self.loss_func = BSANLoss(**kwargs)
        elif loss_type == 'san':
            self.loss_func = SANLoss(**kwargs)
        elif loss_type == 'sladv':
            self.loss_func = SLADVLoss(**kwargs)
        elif loss_type == 'ladv':
            self.loss_func = LADVLoss(**kwargs)
        elif loss_type == 'adv':
            self.loss_func = AdversarialLoss(**kwargs)
        else:
            raise ValueError(f'Unsupported transfer loss: {loss_type}')

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)
