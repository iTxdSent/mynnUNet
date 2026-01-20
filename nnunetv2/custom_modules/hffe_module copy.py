# nnunetv2/custom_modules/hffe_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


class ConvNormNonlin(nn.Module):
    """
    nnUNet-friendly conv block: Conv -> Norm (optional) -> Nonlin (optional)
    """
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        self.conv = conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = norm_op(out_ch, **norm_op_kwargs) if norm_op is not None else None
        self.act = nonlin(**nonlin_kwargs) if nonlin is not None else None
        self.nonlin_first = nonlin_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.nonlin_first:
            if self.act is not None:
                x = self.act(x)
            if self.norm is not None:
                x = self.norm(x)
        else:
            if self.norm is not None:
                x = self.norm(x)
            if self.act is not None:
                x = self.act(x)
        return x


class SpatialAttention(nn.Module):
    """
    CBAM-style SAM: avg/max over channel -> conv -> sigmoid -> multiply
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 5, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att


class CoordAttMap(nn.Module):
    """
    Coordinate Attention that RETURNS attention map (B,C,H,W) in [0,1],
    instead of returning x * att.

    This aligns better with paper Eq.(15) where SWMfuse is a weight map.
    """
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        channels: int,
        reduction: int = 32,
        bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
        temperature: float = 1.0, # wendu,1
        eps: float = 0.0          # eps,0

    ):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}

        mip = max(8, channels // reduction)
        self.conv1 = conv_op(channels, mip, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = norm_op(mip, **norm_op_kwargs) if norm_op is not None else None
        self.act = nonlin(**nonlin_kwargs) if nonlin is not None else nn.ReLU(inplace=True)
        self.nonlin_first = nonlin_first

        self.conv_h = conv_op(mip, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_w = conv_op(mip, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.temperature = temperature
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        



    def _norm_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlin_first:
            x = self.act(x)
            if self.norm is not None:
                x = self.norm(x)
        else:
            if self.norm is not None:
                x = self.norm(x)
            x = self.act(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)              # [B,C,H,1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B,C,W,1]
        y = torch.cat([x_h, x_w], dim=2)               # [B,C,H+W,1]
        y = self._norm_act(self.conv1(y))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)                  # [B,mip,1,W]
        # ---温度控制 T ---
        logit_h = self.conv_h(y_h)
        logit_w = self.conv_w(y_w)
        if self.temperature != 1.0:
            logit_h = logit_h / self.temperature
            logit_w = logit_w / self.temperature
        a_h = self.sigmoid(logit_h)           # [B,C,H,1]
        a_w = self.sigmoid(logit_w)           # [B,C,1,W]
        att = a_h * a_w                                # broadcast to [B,C,H,W]

        if self.eps > 0.0:
            att = self.eps + (1.0 - 2.0 * self.eps) * att

        return att


class HFFE(nn.Module):
    """
    Paper-aligned HFFE (Eq.9-16), with nnUNet-friendly knobs.

    Key defaults:
    - align_corners=False (recommended with nnUNet pipeline)
    - use_residual_gate=False (module EFFECTIVE by default)
    - use_cross_residual=True (Eq.13-14 actually influences Eq.16)
    """
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        c_low: int,
        c_high: int,
        out_ch: Optional[int] = None,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
        sam_kernel: int = 7,
        coord_reduction: int = 32,
        align_corners: bool = False,
        use_cross_residual: bool = True,
        use_residual_gate: bool = False,
        gate_init: float = 1.0,
        split_fuse_gate: bool = False,
        debug: bool = False,
        swm_temperature: float = 1.0,
        swm_eps: float = 0.0,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = c_low

        self.align_corners = align_corners
        self.use_cross_residual = use_cross_residual
        self.use_residual_gate = use_residual_gate
        self.split_fuse_gate = bool(split_fuse_gate)


        # Eq.(9)(10): F' = CBR3(SAM(.))
        self.sam_low = SpatialAttention(kernel_size=sam_kernel)
        self.sam_high = SpatialAttention(kernel_size=sam_kernel)
        self.cbr3_low = ConvNormNonlin(conv_op, c_low,  c_low,  kernel_size=3, bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)
        self.cbr3_high = ConvNormNonlin(conv_op, c_high, c_high, kernel_size=3, bias=conv_bias,
                                        norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                        nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)

        # Eq.(11)(12): SWM = sigmoid(CBR1(F')) -> 1-channel spatial weight
        # NOTE: for nnUNet (InstanceNorm), 1-channel norm can be unstable; allow it but keep simple.
        self.swm_low_conv = ConvNormNonlin(conv_op, c_low, 1, kernel_size=1, padding=0, bias=conv_bias,
                                           norm_op=None, norm_op_kwargs=None,
                                           nonlin=None, nonlin_kwargs=None, nonlin_first=False)
        self.swm_high_conv = ConvNormNonlin(conv_op, c_high, 1, kernel_size=1, padding=0, bias=conv_bias,
                                            norm_op=None, norm_op_kwargs=None,
                                            nonlin=None, nonlin_kwargs=None, nonlin_first=False)
        self.sigmoid = nn.Sigmoid()

        # Eq.(15): SWMfuse = CA( Conv1x1([F'low, F'high]) )   (CA returns map in [0,1])
        self.fuse_pre = ConvNormNonlin(conv_op, c_low + c_high, out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)
        if self.split_fuse_gate:
            self.coord_att_map_low = CoordAttMap(conv_op, out_ch, reduction=coord_reduction, bias=conv_bias,
                                            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first,temperature=swm_temperature,
                                            eps=swm_eps)
            self.coord_att_map_high = CoordAttMap(conv_op, out_ch, reduction=coord_reduction, bias=conv_bias,
                                            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first,temperature=swm_temperature,
                                            eps=swm_eps)
        else:
            self.coord_att_map = CoordAttMap(conv_op, out_ch, reduction=coord_reduction, bias=conv_bias,
                                            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first,temperature=swm_temperature,
                                            eps=swm_eps)

        # Eq.(16): proj -> gate by SWMfuse -> concat -> Conv1x1
        self.proj_low = ConvNormNonlin(conv_op, c_low,  out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)
        self.proj_high = ConvNormNonlin(conv_op, c_high, out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                        norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                        nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)
        # Base passthrough projections (no norm/nonlin): keep information flow even if gate is small
        self.base_low = ConvNormNonlin(conv_op, c_low, out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                        norm_op=None, norm_op_kwargs=None,
                                        nonlin=None, nonlin_kwargs=None, nonlin_first=False)
        self.base_high = ConvNormNonlin(conv_op, c_high, out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                        norm_op=None, norm_op_kwargs=None,
                                        nonlin=None, nonlin_kwargs=None, nonlin_first=False)

        self.out_conv = ConvNormNonlin(conv_op, out_ch * 2, out_ch, kernel_size=1, padding=0, bias=conv_bias,
                                       norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                                       nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first)

        # Optional residual gate controlled by Trainer
        self.register_buffer("gate_scale", torch.tensor(float(gate_init)))
        self.debug = bool(debug)
        self.swm_temperature = float(swm_temperature)
        self.swm_eps = float(swm_eps)
        if not (0.0 <= self.swm_eps < 0.5):
            raise ValueError(f"swm_eps must be in [0, 0.5). Got {self.swm_eps}")
        if self.swm_temperature <= 0:
            raise ValueError(f"swm_temperature must be > 0. Got {self.swm_temperature}")
        self.debug_last = {}

    def set_gate_scale(self, scale: float) -> None:
        self.gate_scale.fill_(float(scale))

    @torch.no_grad()
    def _save_debug(self, swm_low, swm_high, swm_fuse):
        # store lightweight stats only (avoid big CPU tensors by default)
        def _stats(t: torch.Tensor):
            return {
                "min": float(t.min()),
                "max": float(t.max()),
                "mean": float(t.mean()),
                "p01": float(torch.quantile(t.flatten(), 0.01)),
                "p99": float(torch.quantile(t.flatten(), 0.99)),
                "sat_low(<0.05)": float((t < 0.05).float().mean()),
                "sat_high(>0.95)": float((t > 0.95).float().mean()),
            }
        self.debug_last = {
            "swm_low": _stats(swm_low.detach()),
            "swm_high": _stats(swm_high.detach()),
            "swm_fuse": _stats(swm_fuse.detach()),
        }

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # align high -> low resolution
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_high_up = F.interpolate(x_high, size=x_low.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        else:
            x_high_up = x_high

        # Eq.(9)(10)
        f_low_p = self.cbr3_low(self.sam_low(x_low))
        f_high_p = self.cbr3_high(self.sam_high(x_high_up))

        # Eq.(11)(12)
        swm_low = self.sigmoid(self.swm_low_conv(f_low_p))       # [B,1,H,W]
        swm_high = self.sigmoid(self.swm_high_conv(f_high_p))    # [B,1,H,W]

        # Eq.(13)(14)
        f_low_pp = x_low * swm_high
        f_high_pp = x_high_up * swm_low

        # Eq.(15) -> map in [0,1]（旧版本未做拼接，已修改）
        fuse_mid = self.fuse_pre(torch.cat([f_low_p + f_low_pp, f_high_p + f_high_pp], dim=1))

        #swm_fuse = self.coord_att_map(fuse_mid)  # [B,out,H,W] in [0,1]加了一个温度控制一下
        # 原版：
        #low_term = swm_fuse * self.proj_low(low_base)
        #high_term = swm_fuse * self.proj_high(high_base)
        # ---- Eq.(15) CoordAtt -> SWM_fuse with optional temperature/eps-floor ----

        if self.split_fuse_gate:
            g_low  = self.coord_att_map_low(fuse_mid)
            g_high = self.coord_att_map_high(fuse_mid)
            swm_fuse = 0.5*(g_low + g_high) #debug only
        else:
            swm_fuse = self.coord_att_map(fuse_mid)
            g_low = swm_fuse
            g_high = swm_fuse
        
        # Eq.(16)
        # base passthrough (NOT gated)
        base_low  = self.base_low(x_low)
        base_high = self.base_high(x_high_up)




        # enhanced branch (gated)
        if self.use_cross_residual:
            enh_low  = self.proj_low(f_low_p + f_low_pp)
            enh_high = self.proj_high(f_high_p + f_high_pp)
        else:
            enh_low  = self.proj_low(f_low_p)      # base already contains x_low
            enh_high = self.proj_high(f_high_p)    # base already contains x_high_up

        low_term  = base_low  + g_low * enh_low
        high_term = base_high + g_high * enh_high

        out = self.out_conv(torch.cat([low_term, high_term], dim=1))

        #origin：




        if self.debug:
            self._save_debug(swm_low, swm_high, swm_fuse)

        if self.use_residual_gate:
            # stable warmup: out = x_low + s*(out - x_low)
            #s = float(self.gate_scale.item())
            s = self.gate_scale.to(dtype=out.dtype, device=out.device).clamp_(0.0, 1.0)
            out = x_low + s * (out - x_low)

        return out


def build_hffe_pyramid(
    conv_op: Type[_ConvNd],
    features_per_stage: Sequence[int],
    conv_bias: bool,
    norm_op: Optional[Type[nn.Module]],
    norm_op_kwargs: Optional[dict],
    nonlin: Optional[Type[nn.Module]],
    nonlin_kwargs: Optional[dict],
    nonlin_first: bool,
    split_fuse_gate: bool = False,
    swm_temperature: float = 1.0,
    swm_eps: float = 0.0,
    **hffe_kwargs,
) -> nn.ModuleList:
    """
    Create HFFE modules for (E0,E1)...(E_{n-2},E_{n-1}).
    Output channels default to c_low so we can REPLACE skip feature without changing decoder.
    """
    hffes = []
    for i in range(len(features_per_stage) - 1):
        c_low = int(features_per_stage[i])
        c_high = int(features_per_stage[i + 1])
        hffes.append(
            HFFE(
                conv_op=conv_op,
                c_low=c_low,
                c_high=c_high,
                out_ch=c_low,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                nonlin_first=nonlin_first,
                split_fuse_gate=split_fuse_gate,
                swm_temperature=swm_temperature,
                swm_eps=swm_eps,
                **hffe_kwargs,
            )
        )
    return nn.ModuleList(hffes)
