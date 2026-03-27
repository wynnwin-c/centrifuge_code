# 2_models/pdan_selective.py
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_loss import (
    GradientReversal,
    binary_domain_loss,
    cross_entropy,
    class_probability,
    class_balanced_weights_from_target,
)


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


class WeightedDomainDiscriminator(nn.Module):
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


class PDANSelective(nn.Module):
    """
    部分域迁移：对抗对齐时按类权重抑制非共享源类。
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_dim: int = 256,
        init_class_weights: Optional[torch.Tensor] = None,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.feature_extractor = Conv1DBackbone(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = ClassifierHead(feat_dim, num_classes)
        self.domain_discriminator = WeightedDomainDiscriminator(feat_dim)
        self.grl = GradientReversal(1.0)
        print("======== 拦截到的初始权重是：========", init_class_weights)
        self.register_buffer("class_weights", torch.ones(num_classes))
        if init_class_weights is not None:
            self.class_weights.copy_(init_class_weights.clamp_min(1e-6))
        self.momentum = momentum

    @torch.no_grad()
    def update_class_weights(self, target_logits: torch.Tensor):
        w = class_balanced_weights_from_target(target_logits, momentum=self.momentum, prev_w=self.class_weights)
        self.class_weights.copy_(w.clamp_min(1e-6))

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        return self.classifier(f)

    def selective_domain_weight(
        self,
        src_logits: Optional[torch.Tensor],
        tgt_logits: Optional[torch.Tensor],
        detach: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        w_src = None
        w_tgt = None
        if src_logits is not None:
            ps = class_probability(src_logits)  # [Bs, C]
            cw = self.class_weights
            w_src = (ps * cw.unsqueeze(0)).sum(dim=1)  # [Bs]
            if detach:
                w_src = w_src.detach()
        if tgt_logits is not None:
            pt = class_probability(tgt_logits)
            cw = self.class_weights
            w_tgt = (pt * cw.unsqueeze(0)).sum(dim=1)
            if detach:
                w_tgt = w_tgt.detach()
        return w_src, w_tgt

    def forward(
        self,
        src_x: Optional[torch.Tensor] = None,
        tgt_x: Optional[torch.Tensor] = None,
        src_y: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        adapt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        f_s = None
        src_logits = None
        if src_x is not None:
            f_s = self.feature_extractor(src_x)
            src_logits = self.classifier(f_s)
            out["src_logits"] = src_logits
            if src_y is not None:
                out["cls_loss"] = cross_entropy(src_logits, src_y)

        f_t = None
        tgt_logits = None
        if tgt_x is not None:
            f_t = self.feature_extractor(tgt_x)
            tgt_logits = self.classifier(f_t)
            out["tgt_logits"] = tgt_logits

        if adapt and tgt_logits is not None:
            self.update_class_weights(tgt_logits)

        feats = []
        dom = []
        weights = []
        if f_s is not None:
            feats.append(f_s)
            dom.append(torch.zeros(f_s.size(0), device=f_s.device))
        if f_t is not None:
            feats.append(f_t)
            dom.append(torch.ones(f_t.size(0), device=f_t.device))
        if feats:
            feats_cat = torch.cat(feats, dim=0)
            dom_cat = torch.cat(dom, dim=0)
            self.grl.set_lambda(grl_lambda)
            d_logits = self.domain_discriminator(self.grl(feats_cat))
            if src_logits is not None or tgt_logits is not None:
                w_src, w_tgt = self.selective_domain_weight(src_logits, tgt_logits, detach=True)
                if w_src is not None and w_tgt is not None:
                    w = torch.cat([w_src, w_tgt], dim=0)
                elif w_src is not None:
                    w = w_src
                else:
                    w = w_tgt
            else:
                w = None
            out["domain_logits"] = d_logits.squeeze(-1)
            out["domain_loss"] = binary_domain_loss(d_logits, dom_cat, weight=w)
        return out


def build_pdan_selective(in_channels: int, num_classes: int, feat_dim: int = 256) -> PDANSelective:
    return PDANSelective(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)
