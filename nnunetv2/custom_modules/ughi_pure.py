# nnunetv2/custom_modules/ughi_pure_lite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


@dataclass
class PureUGHILiteConfig:
    # 是否使用更深一层的 edge gate
    use_edge_gate: bool = True
    # 是否对 edge gate 做 detach（强烈建议 True，显存/图依赖关键）
    detach_prev_edge_gate: bool = True
    # 注入强度初值（可训练标量）
    inject_scale_init: float = 0.10
    # 上采样时 align_corners
    align_corners: bool = False


class PureUGHILite(nn.Module):
    """
    纯粹 UGHI（Lite版，显存友好）：
    - 只增强竖向 upsample 特征 x_up
    - 语义门控用 1-channel spatial gate（避免 B×C×H×W attention map）
    - prev edge gate 默认 detach，避免把 deep edge head 接入主分割反传路径
    """

    def __init__(
        self,
        conv_op: Type[_ConvNd],
        up_ch: int,
        hffe_ch: int,
        cfg: Optional[PureUGHILiteConfig] = None,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else PureUGHILiteConfig()

        self.h_proj = conv_op(hffe_ch, up_ch, kernel_size=1, stride=1, padding=0, bias=conv_bias)
        self.sem_gate = conv_op(up_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)  # spatial gate
        self.inject_scale = nn.Parameter(torch.tensor(float(self.cfg.inject_scale_init), dtype=torch.float32))

    def forward(
        self,
        x_up: torch.Tensor,                        # (B, C_up, H, W)
        hffe_feat: Optional[torch.Tensor],         # (B, C_h, H', W')
        prev_edge_gate: Optional[torch.Tensor],    # (B, 1, h, w) already sigmoid-ed (recommended)
    ) -> torch.Tensor:
        if hffe_feat is None:
            return x_up

        # spatial align
        if hffe_feat.shape[-2:] != x_up.shape[-2:]:
            h = F.interpolate(
                hffe_feat,
                size=x_up.shape[-2:],
                mode="bilinear",
                align_corners=bool(self.cfg.align_corners),
            )
        else:
            h = hffe_feat

        # channel align
        h = self.h_proj(h)  # (B, C_up, H, W)

        # semantic spatial gate: (B,1,H,W)
        s = torch.sigmoid(self.sem_gate(h))

        # edge gate (already prob map). interpolate + (optional) detach
        if self.cfg.use_edge_gate and (prev_edge_gate is not None):
            g = prev_edge_gate
            if g.shape[-2:] != x_up.shape[-2:]:
                g = F.interpolate(g, size=x_up.shape[-2:], mode="bilinear", align_corners=False)
            if self.cfg.detach_prev_edge_gate:
                g = g.detach()
        else:
            g = 1.0

        inj = (g * s) * h  # (B,C,H,W)
        return x_up + self.inject_scale * inj
