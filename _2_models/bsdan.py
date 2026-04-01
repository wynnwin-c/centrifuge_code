# 2_models/bsdan.py
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_loss import GradientReversal, cross_entropy

class Conv1DBackbone(nn.Module):
    """保持与你原有模型完全一致的特征提取器 (1D-CNN)"""
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
    """保持与你原有模型一致的分类头"""
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


class SubDomainDiscriminator(nn.Module):
    """【BSDAN创新】子领域判别器：每个类别专属一个"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1), # 注意：这里不加Sigmoid，使用BCEWithLogitsLoss更稳定
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class BSDAN(nn.Module):
    """
    Balanced Selective Domain Adversarial Network (BSDAN)
    包含：子领域判别器、目标域分布类级加权、实例级加权、目标域熵最小化、源域扩充策略。
    """
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = Conv1DBackbone(in_channels=in_channels, feat_dim=feat_dim)
        self.classifier = ClassifierHead(feat_dim, num_classes)
        
        # 实例化 C 个子领域判别器
        self.sub_discriminators = nn.ModuleList([
            SubDomainDiscriminator(feat_dim) for _ in range(num_classes)
        ])
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
        rho: float = 0.0 # 【BSDAN创新参数】: 平衡扩充比例
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        
        # 1. 源域特征提取与分类
        f_s, src_logits = None, None
        if src_x is not None:
            f_s = self.feature_extractor(src_x)
            src_logits = self.classifier(f_s)
            out["src_logits"] = src_logits
            if src_y is not None:
                out["cls_loss"] = cross_entropy(src_logits, src_y)

        # 2. 目标域特征提取与分类（摸底测验）
        f_t, tgt_logits = None, None
        if tgt_x is not None:
            f_t = self.feature_extractor(tgt_x)
            tgt_logits = self.classifier(f_t)
            out["tgt_logits"] = tgt_logits
            
            # 【BSDAN创新】：目标域熵最小化损失 (Eq 4-4)
            # 强迫模型对无标签的目标域数据更加自信
            tgt_probs_ent = F.softmax(tgt_logits, dim=-1)
            entropy_loss = - (tgt_probs_ent * torch.log(tgt_probs_ent + 1e-8)).sum(dim=-1).mean()
            out["entropy_loss"] = entropy_loss

        # 3. 加权子领域对抗损失 (Eq 4-1, 4-2, 4-5 融合)
        if f_s is not None and f_t is not None:
            # 获取概率并脱离计算图 (防止判别器的梯度影响分类器)
            src_probs = F.softmax(src_logits, dim=-1).detach() # p_i^c
            tgt_probs = F.softmax(tgt_logits, dim=-1).detach() # p_j^c
            
            # 【BSDAN创新】：正确的类级权重！(Eq 4-2) 目标域不存在的类，权重就是0！
            class_weights = tgt_probs.mean(dim=0) # [C]
            
            self.grl.set_lambda(grl_lambda)
            total_domain_loss = 0.0
            
            for c in range(self.num_classes):
                # 提取当前类别 c 的权重和概率
                w_c = class_weights[c]
                p_c_s = src_probs[:, c:c+1] # [Ns, 1]
                p_c_t = tgt_probs[:, c:c+1] # [Nt, 1]
                
                # 【BSDAN创新】：实例级特征加权 (用概率作为门控，只放行该类别的特征)
                feat_s_c = self.grl(f_s * p_c_s)
                feat_t_c = self.grl(f_t * p_c_t)
                
                # 送入专属于类别 c 的子判别器
                logits_d_s = self.sub_discriminators[c](feat_s_c)
                logits_d_t = self.sub_discriminators[c](feat_t_c)
                
                # 生成标签：源域为1，目标域为0
                label_s = torch.ones_like(logits_d_s)
                label_t = torch.zeros_like(logits_d_t)
                
                # 计算 BCE Loss
                loss_s = F.binary_cross_entropy_with_logits(logits_d_s, label_s)
                loss_t = F.binary_cross_entropy_with_logits(logits_d_t, label_t)
                
                # 【BSDAN极其优雅的扩充策略】 (Eq 4-5 第3项)
                # 不需要修改DataLoader！直接让判别器把源域特征当成目标域(label=0)来判别，并乘上扩充比例rho！
                loss_exp = F.binary_cross_entropy_with_logits(logits_d_s, torch.zeros_like(logits_d_s))
                
                # 累加当前类别的 Loss (类级权重 * (源损失 + 目标损失 + 扩充伪装损失))
                total_domain_loss += w_c * (loss_s + loss_t + rho * loss_exp)
                
            out["domain_loss"] = total_domain_loss
            
        return out


def build_bsdan(in_channels: int, num_classes: int, feat_dim: int = 256) -> BSDAN:
    return BSDAN(in_channels=in_channels, num_classes=num_classes, feat_dim=feat_dim)