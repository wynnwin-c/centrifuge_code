# 2_models/utils_loss.py
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambd)

    def set_lambda(self, v: float):
        self.lambd = float(v)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0.0:
        n = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(label_smoothing / (n - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        return torch.mean(torch.sum(-true_dist * logp, dim=-1))
    return F.cross_entropy(logits, targets)


def binary_domain_loss(domain_logits: torch.Tensor, domain_labels: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(domain_logits.squeeze(-1), domain_labels.float(), weight=weight)


def entropy_minimization(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return torch.sum(-p * torch.log(p.clamp_min(1e-8)), dim=-1).mean()


def softmax_entropy_weight(logits: torch.Tensor, detach: bool = True) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    ent = -torch.sum(p * torch.log(p.clamp_min(1e-8)), dim=-1)
    w = 1.0 + torch.exp(-ent)
    return w.detach() if detach else w


def class_probability(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    return F.softmax(logits, dim=-1)


def class_balanced_weights_from_target(
    target_logits: torch.Tensor,
    momentum: float = 0.9,
    prev_w: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    with torch.no_grad():
        probs = F.softmax(target_logits, dim=-1)  # [Bt, C]
        freq = probs.mean(0)                      # [C]
        freq = freq / (freq.sum() + eps)
        inv = 1.0 / (freq + eps)
        inv = inv / inv.max().clamp_min(eps)
        if prev_w is not None:
            inv = momentum * prev_w + (1 - momentum) * inv
        inv = inv / inv.max().clamp_min(eps)
        return inv


def maximum_mean_discrepancy(x: torch.Tensor, y: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: Optional[float] = None) -> torch.Tensor:
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        n = total.size(0)
        bandwidth = torch.sum(L2_distance.detach()) / (n ** 2 - n)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    kernel_val = sum(kernels)
    n_x = x.size(0)
    n_y = y.size(0)
    Kxx = kernel_val[:n_x, :n_x]
    Kyy = kernel_val[n_x:, n_x:]
    Kxy = kernel_val[:n_x, n_x:]
    Kyx = kernel_val[n_x:, :n_x]
    return torch.mean(Kxx) + torch.mean(Kyy) - torch.mean(Kxy) - torch.mean(Kyx)


def consistency_loss(p: torch.Tensor, q: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    kl = F.kl_div(p, q, reduction="none").sum(-1)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0, learn_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.scale = nn.Parameter(torch.tensor(float(scale))) if learn_scale else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        logits = torch.matmul(x_norm, w_norm.t())
        if self.scale is not None:
            logits = logits * self.scale
        return logits


def mixup_features(x: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        lam = 1.0
    else:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(x.device)
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[index]
    return x_mix, index


def build_optimizer(model: nn.Module, lr: float = 1e-3, wd: float = 1e-4, betas=(0.9, 0.999)) -> torch.optim.Optimizer:
    params: Dict[str, list] = {"backbone": [], "head": [], "discriminator": [], "others": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "feature_extractor" in n or "backbone" in n:
            params["backbone"].append(p)
        elif "classifier" in n or "fc" in n or "head" in n:
            params["head"].append(p)
        elif "domain_discriminator" in n or "discriminator" in n:
            params["discriminator"].append(p)
        else:
            params["others"].append(p)
    groups = []
    if params["backbone"]:
        groups.append({"params": params["backbone"], "lr": lr * 0.5, "weight_decay": wd})
    if params["head"]:
        groups.append({"params": params["head"], "lr": lr, "weight_decay": wd})
    if params["discriminator"]:
        groups.append({"params": params["discriminator"], "lr": lr, "weight_decay": wd})
    if params["others"]:
        groups.append({"params": params["others"], "lr": lr, "weight_decay": wd})
    if not groups:
        groups = [{"params": model.parameters(), "lr": lr, "weight_decay": wd}]
    opt = torch.optim.Adam(groups, lr=lr, betas=betas, weight_decay=wd)
    return opt
