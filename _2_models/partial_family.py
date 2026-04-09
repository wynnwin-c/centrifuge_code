from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_loss import GradientReversal, cross_entropy


class Conv1DBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, feat_dim: int = 256, use_bn: bool = True):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c, 2 * c, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(2 * c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * c, 2 * c, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2 * c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(2 * c, 4 * c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4 * c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(4 * c, 4 * c, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(4 * c) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LocalDomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PartialAdversarialModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_dim: int = 256,
        selective: bool = False,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.selective = selective
        self.entropy_weight = entropy_weight
        self.feature_extractor = Conv1DBackbone(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = ClassifierHead(feat_dim, num_classes)
        self.local_discriminators = nn.ModuleList([LocalDomainDiscriminator(feat_dim) for _ in range(num_classes)])
        self.grl = GradientReversal(1.0)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        return self.classifier(f)

    def _class_weights(self, target_probs: torch.Tensor) -> torch.Tensor:
        if not self.selective:
            return torch.ones(self.num_classes, device=target_probs.device)
        weights = target_probs.mean(dim=0)
        return weights / weights.max().clamp_min(1e-6)

    def forward(
        self,
        src_x: Optional[torch.Tensor] = None,
        tgt_x: Optional[torch.Tensor] = None,
        src_y: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        f_s, src_logits = None, None
        if src_x is not None:
            f_s = self.feature_extractor(src_x)
            src_logits = self.classifier(f_s)
            out["src_logits"] = src_logits
            if src_y is not None:
                out["cls_loss"] = cross_entropy(src_logits, src_y)

        f_t, tgt_logits = None, None
        if tgt_x is not None:
            f_t = self.feature_extractor(tgt_x)
            tgt_logits = self.classifier(f_t)
            out["tgt_logits"] = tgt_logits

        if f_s is not None and f_t is not None:
            src_probs = F.softmax(src_logits, dim=-1)
            tgt_probs = F.softmax(tgt_logits, dim=-1)
            class_weights = self._class_weights(tgt_probs)
            self.grl.set_lambda(grl_lambda)
            total_domain_loss = torch.tensor(0.0, device=f_s.device)

            for c in range(self.num_classes):
                src_gate = src_probs[:, c:c+1]
                tgt_gate = tgt_probs[:, c:c+1]
                feat_s_c = self.grl(f_s * src_gate)
                feat_t_c = self.grl(f_t * tgt_gate)
                logits_d_s = self.local_discriminators[c](feat_s_c)
                logits_d_t = self.local_discriminators[c](feat_t_c)
                label_s = torch.ones_like(logits_d_s)
                label_t = torch.zeros_like(logits_d_t)
                loss_s = F.binary_cross_entropy_with_logits(logits_d_s, label_s)
                loss_t = F.binary_cross_entropy_with_logits(logits_d_t, label_t)
                weight = class_weights[c] if self.selective else 1.0
                total_domain_loss = total_domain_loss + weight * 0.5 * (loss_s + loss_t)

            out["domain_loss"] = total_domain_loss

            if self.entropy_weight > 0.0:
                entropy_loss = -(tgt_probs * torch.log(tgt_probs.clamp_min(1e-8))).sum(dim=-1).mean()
                out["entropy_loss"] = entropy_loss

        return out


class LADVModel(PartialAdversarialModel):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim, selective=False, entropy_weight=0.0)


class SLADVModel(PartialAdversarialModel):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim, selective=True, entropy_weight=0.0)


class SANModel(PartialAdversarialModel):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim, selective=True, entropy_weight=0.1)


class BSANModel(PartialAdversarialModel):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim, selective=True, entropy_weight=0.1)


def build_ladv(in_channels: int, num_classes: int, feat_dim: int = 256) -> LADVModel:
    return LADVModel(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)


def build_sladv(in_channels: int, num_classes: int, feat_dim: int = 256) -> SLADVModel:
    return SLADVModel(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)


def build_san(in_channels: int, num_classes: int, feat_dim: int = 256) -> SANModel:
    return SANModel(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)


def build_bsan(in_channels: int, num_classes: int, feat_dim: int = 256) -> BSANModel:
    return BSANModel(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)
