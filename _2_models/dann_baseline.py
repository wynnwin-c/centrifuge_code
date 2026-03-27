# 2_models/dann_baseline.py
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_loss import GradientReversal, binary_domain_loss, cross_entropy


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
        x = self.conv(x)
        x = self.proj(x)
        return x


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


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class DANN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__()
        self.feature_extractor = Conv1DBackbone(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = ClassifierHead(feat_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(feat_dim)
        self.grl = GradientReversal(1.0)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        return self.classifier(f)

    def forward(
        self,
        src_x: Optional[torch.Tensor] = None,
        tgt_x: Optional[torch.Tensor] = None,
        src_y: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if src_x is not None:
            f_s = self.feature_extractor(src_x)
            cls_logits = self.classifier(f_s)
            out["src_logits"] = cls_logits
            if src_y is not None:
                out["cls_loss"] = cross_entropy(cls_logits, src_y)
        else:
            f_s = None

        feats_for_domain = []
        dom_labels = []
        if f_s is not None:
            feats_for_domain.append(f_s)
            dom_labels.append(torch.zeros(f_s.size(0), device=f_s.device))
        if tgt_x is not None:
            f_t = self.feature_extractor(tgt_x)
            feats_for_domain.append(f_t)
            dom_labels.append(torch.ones(f_t.size(0), device=f_t.device))
            out["tgt_feats"] = f_t
        if feats_for_domain:
            feats = torch.cat(feats_for_domain, dim=0)
            labels = torch.cat(dom_labels, dim=0)
            self.grl.set_lambda(grl_lambda)
            d_logits = self.domain_discriminator(self.grl(feats))
            out["domain_logits"] = d_logits.squeeze(-1)
            out["domain_loss"] = binary_domain_loss(d_logits, labels)
        return out


def build_dann(in_channels: int, num_classes: int, feat_dim: int = 256) -> DANN:
    return DANN(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)
