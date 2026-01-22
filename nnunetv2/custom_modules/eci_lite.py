# nnunetv2/custom_modules/eci_lite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


def mask_to_edge_band_torch(mask: torch.Tensor, radius: int = 1, *, assume_binary: bool = True) -> torch.Tensor:
    """
    Generate a binary "edge band" from a binary mask via morphological gradient:
        edge = dilate(mask) - erode(mask)

    Pure torch (MaxPool-based). Supports shapes:
      (H,W) | (B,H,W) | (B,1,H,W)

    Returns the same rank as input (float tensor in [0,1]).
    """
    if radius <= 0:
        raise ValueError("radius must be >= 1")

    if mask.ndim == 2:
        m = mask[None, None]
        squeeze_back = 2
    elif mask.ndim == 3:
        m = mask[:, None]
        squeeze_back = 3
    elif mask.ndim == 4:
        m = mask
        squeeze_back = 4
    else:
        raise ValueError(f"mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    if m.shape[1] != 1:
        raise ValueError(f"mask channel must be 1, got C={m.shape[1]}")

    m = m.float()
    if assume_binary:
        m = (m > 0.5).float()

    k = 2 * radius + 1
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    ero = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=radius)
    edge = (dil - ero).clamp(0.0, 1.0)

    if squeeze_back == 2:
        return edge[0, 0]
    if squeeze_back == 3:
        return edge[:, 0]
    return edge


# [NEW] SE Block Implementation
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Ensure minimal channels for reduction to avoid 0
        mid_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


@dataclass
class ECILiteConfig:
    """
    ECI-Lite: Edge-Cue Interaction (lightweight).
    """
    edge_ratio: float = 0.125
    edge_min_channels: int = 8
    use_depthwise: bool = True
    edge_kernel_size: int = 3
    inject_scale_init: float = 0.0
    detach_edge_loss: bool = True
    return_edge_logits_default: bool = False
    use_se: bool = False  # [NEW] Enable Squeeze-and-Excitation by default


class ECILite(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        *,
        cfg: Optional[ECILiteConfig] = None,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv2d:
            raise NotImplementedError("ECILite currently supports Conv2d only (2D).")
        if channels <= 0:
            raise ValueError("channels must be > 0")

        self.conv_op = conv_op
        self.channels = int(channels)
        self.cfg = cfg if cfg is not None else ECILiteConfig()

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        # nnUNet-like defaults
        if norm_op is None:
            norm_op = nn.InstanceNorm2d
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin is None:
            nonlin = nn.LeakyReLU
            nonlin_kwargs = {"inplace": True}

        edge_ch = max(self.cfg.edge_min_channels, int(round(self.channels * float(self.cfg.edge_ratio))))
        self.edge_channels = int(edge_ch)

        k = int(self.cfg.edge_kernel_size)
        if k % 2 != 1:
            raise ValueError("edge_kernel_size must be odd (e.g., 3).")
        p = k // 2

        # edge feature extractor (lightweight)
        if self.cfg.use_depthwise:
            self.edge_dw = conv_op(self.channels, self.channels, kernel_size=k, padding=p,
                                   groups=self.channels, bias=False)
            self.edge_dw_norm = norm_op(self.channels, **norm_op_kwargs)
            self.edge_pw = conv_op(self.channels, self.edge_channels, kernel_size=1, padding=0, bias=conv_bias)
            self.edge_pw_norm = norm_op(self.edge_channels, **norm_op_kwargs)
            self.edge_act = nonlin(**nonlin_kwargs)
            self._edge_mode = "depthwise"
        else:
            self.edge_conv = conv_op(self.channels, self.edge_channels, kernel_size=k, padding=p, bias=conv_bias)
            self.edge_norm = norm_op(self.edge_channels, **norm_op_kwargs)
            self.edge_act = nonlin(**nonlin_kwargs)
            self._edge_mode = "plain"
        
        # [NEW] Initialize SE Block
        if self.cfg.use_se:
            self.se = SEBlock(self.edge_channels, reduction=4)

        # gates
        self.main_gate = conv_op(self.channels, 1, kernel_size=1, padding=0, bias=True)
        self.edge_gate = conv_op(self.edge_channels, 1, kernel_size=1, padding=0, bias=True)

        # projection + auxiliary head
        self.edge_proj = conv_op(self.edge_channels, self.channels, kernel_size=1, padding=0, bias=True)
        self.edge_head = conv_op(self.edge_channels, 1, kernel_size=1, padding=0, bias=True)

        # Whether to compute edge logits head. Trainer may disable this when edge loss is off.
        # When disabled, the injection path remains active (uses edge features + gates),
        # but edge_logits will not be produced.
        self.edge_head_enabled: bool = True

        # runtime scaling buffer
        self.register_buffer("inject_scale", torch.tensor(float(self.cfg.inject_scale_init), dtype=torch.float32))

        self.debug_last: Dict[str, Any] = {}

    def set_inject_scale(self, s: float) -> None:
        self.inject_scale.fill_(float(s))

    def set_edge_head_enabled(self, enabled: bool) -> None:
        self.edge_head_enabled = bool(enabled)

    def set_edge_head_enabled(self, enabled: bool) -> None:
        self.edge_head_enabled = bool(enabled)

    def _edge_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._edge_mode == "depthwise":
            e = self.edge_dw(x)
            e = self.edge_dw_norm(e)
            e = self.edge_act(e)
            e = self.edge_pw(e)
            e = self.edge_pw_norm(e)
            e = self.edge_act(e)
            return e
        e = self.edge_conv(x)
        e = self.edge_norm(e)
        e = self.edge_act(e)
        return e

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_edge_logits: Optional[bool] = None,
        store_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected x as (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Channel mismatch: module C={self.channels}, input C={x.shape[1]}")

        if return_edge_logits is None:
            return_edge_logits = bool(self.cfg.return_edge_logits_default)

        if bool(return_edge_logits) and (not bool(self.edge_head_enabled)):
            raise RuntimeError(
                "return_edge_logits=True but edge_head is disabled. Enable edge head or call with return_edge_logits=False."
            )

        edge_feat = self._edge_features(x)
        # Apply SE Block before gating (Channel Attention first)
        if self.cfg.use_se:
            edge_feat = self.se(edge_feat)

        edge_feat_inj = edge_feat

        U = 1.0 - torch.sigmoid(self.main_gate(x))             # (B,1,H,W)
        A = torch.sigmoid(self.edge_gate(edge_feat_inj))       # (B,1,H,W)

        inj = (U * A) * self.edge_proj(edge_feat_inj)          # (B,C,H,W)
        

        y = x + self.inject_scale.to(dtype=x.dtype, device=x.device) * inj

        edge_logits = None
        if bool(self.edge_head_enabled):
            # Only compute edge logits when enabled. This can be disabled to save compute/memory
            # when edge loss is turned off in the trainer.
            edge_logits = self.edge_head(edge_feat.detach() if self.cfg.detach_edge_loss else edge_feat)

        if store_debug:
            with torch.no_grad():
                self.debug_last = {
                    "U_mean": float(U.mean().item()),
                    "A_mean": float(A.mean().item()),
                    "inj_abs_mean": float(inj.abs().mean().item()),
                    "inject_scale": float(self.inject_scale.item()),
                }

        if return_edge_logits:
            return y, edge_logits
        return y
def build_eci_lite_pyramid(
    *,
    features_per_stage: Sequence[int],
    conv_op: Type[_ConvNd],
    cfg: Optional[ECILiteConfig] = None,
    conv_bias: bool = True,
    norm_op: Optional[Type[nn.Module]] = None,
    norm_op_kwargs: Optional[dict] = None,
    nonlin: Optional[Type[nn.Module]] = None,
    nonlin_kwargs: Optional[dict] = None,
) -> nn.ModuleList:
    """
    Minimal compatibility helper for PlainConvUNetWithECI.

    Build ECILite modules for each decoder stage (n_stages-1), ordered:
      low-res (closest bottleneck) -> high-res.

    For standard nnUNet PlainConvUNet channel pyramid:
      decoder stage output channels = [features_per_stage[-2], features_per_stage[-3], ..., features_per_stage[0]]
    """
    if features_per_stage is None or len(features_per_stage) < 2:
        raise ValueError("features_per_stage must have length >= 2")

    # decoder stages: low-res -> high-res
    channels_per_decoder_stage = list(features_per_stage[-2::-1])

    modules = []
    for c in channels_per_decoder_stage:
        modules.append(
            ECILite(
                conv_op=conv_op,
                channels=int(c),
                cfg=cfg,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        )
    return nn.ModuleList(modules)