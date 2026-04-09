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
    def __init__(self, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter, **_ignored):
        super().__init__()
        self.domain_classifier = AdversarialNet(input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter)

    def get_adversarial_result(self, x, source=True):
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        domain_label = torch.ones(len(x), 1, device=device) if source else torch.zeros(len(x), 1, device=device)
        return nn.BCELoss()(domain_pred, domain_label)

    def forward(self, source, target):
        return 0.5 * (self.get_adversarial_result(source, True) + self.get_adversarial_result(target, False))


class EDANNLoss(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter, **_ignored):
        super().__init__()
        self.domain_classifier = AdversarialNet(input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter)
        self.iter_num = 0
        self.high = 1.0
        self.low = 0.0
        self.alpha = 10
        self.max_iter = max_iter

    def entropy(self, input_):
        entropy = -input_ * torch.log(input_ + 1e-7)
        return torch.sum(entropy, dim=1)

    def entropy_loss(self, input_):
        mask = input_.ge(1e-6)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def get_adversarial_result(self, x, weight, source=True):
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        domain_label = torch.ones(len(x), 1, device=device) if source else torch.zeros(len(x), 1, device=device)
        losses = nn.BCELoss(reduction='none')(domain_pred, domain_label)
        return torch.sum(weight * losses) / (1e-8 + torch.sum(weight).detach().item())

    def forward(self, source, target, source_logits, target_logits):
        source_entropy = self.entropy(source_logits)
        target_entropy = self.entropy(target_logits)
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        source_entropy.register_hook(grl_hook(coeff))
        target_entropy.register_hook(grl_hook(coeff))
        source_entropy = 1.0 + torch.exp(-source_entropy)
        target_entropy = 1.0 + torch.exp(-target_entropy)
        source_entropy = (source_entropy / torch.sum(source_entropy).detach().item()).view(-1, 1)
        target_entropy = (target_entropy / torch.sum(target_entropy).detach().item()).view(-1, 1)
        source_loss = self.get_adversarial_result(source, source_entropy, True)
        target_loss = self.get_adversarial_result(target, target_entropy, False)
        return 0.5 * (source_loss + target_loss) + 0.1 * self.entropy_loss(target_logits)


class MMDLoss(nn.Module):
    def __init__(self, **_ignored):
        super().__init__()

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def dan(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        return torch.mean(xx + yy - xy - yx)

    def forward(self, source, target):
        return self.dan(source, target)


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
        mask = input_.ge(1e-6)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def forward(self, source, target, source_logits, target_logits):
        local_loss = super().forward(source, target, source_logits, target_logits)
        return local_loss + 0.1 * self.entropy_loss(target_logits)


class BSANLoss(SLADVLoss):
    def __init__(self, num_class, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter, shared_classes=None):
        super().__init__(num_class, input_num, hidden_num1, hidden_num2, trade_off_adversarial, lam_adversarial, max_iter)
        self.shared_classes = list(shared_classes or [])

    def entropy_loss(self, input_):
        mask = input_.ge(1e-6)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def shared_entropy_loss(self, target_logits):
        if not self.shared_classes:
            return self.entropy_loss(target_logits)
        shared_logits = target_logits[:, self.shared_classes]
        shared_logits = shared_logits / shared_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return self.entropy_loss(shared_logits)

    def forward(self, source, target, source_logits, target_logits, len_share):
        local_loss = super().forward(source, target, source_logits, target_logits)
        if len_share > 0:
            target_logits = target_logits[:source_logits.size(0)]
        return local_loss + 0.1 * self.shared_entropy_loss(target_logits)


class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'bsan':
            self.loss_func = BSANLoss(**kwargs)
        elif loss_type == 'bsdan':
            self.loss_func = BSANLoss(**kwargs)
        elif loss_type == 'san':
            self.loss_func = SANLoss(**kwargs)
        elif loss_type == 'sladv':
            self.loss_func = SLADVLoss(**kwargs)
        elif loss_type == 'ladv':
            self.loss_func = LADVLoss(**kwargs)
        elif loss_type == 'adv':
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == 'edann':
            self.loss_func = EDANNLoss(**kwargs)
        elif loss_type == 'mmd':
            self.loss_func = MMDLoss()
        else:
            raise ValueError(f'Unsupported transfer loss: {loss_type}')

    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)
