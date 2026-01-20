from __future__ import annotations

from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


# 复用你原文件中的 ConvNormNonlin / SpatialAttention
# ---------------------------------------------------
# class ConvNormNonlin(nn.Module): ...
# class SpatialAttention(nn.Module): ...


class CoordAttMapPaper(nn.Module):
    """
    Paper-style Coordinate Attention that RETURNS an attention map (B,C,H,W) in [0,1].
    No temperature/eps tricks; keep it close to standard CoordAtt.

    This is used for Eq.(15): SWM_fuse = sigma( CA( Conv1x1([F'low, F'high]) ) )
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

        # coordinate pooling
        x_h = x.mean(dim=3, keepdim=True)                       # [B,C,H,1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # [B,C,W,1]

        y = torch.cat([x_h, x_w], dim=2)                        # [B,C,H+W,1]
        y = self._norm_act(self.conv1(y))                       # [B,mip,H+W,1]

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)                           # [B,mip,1,W]

        a_h = self.sigmoid(self.conv_h(y_h))                    # [B,C,H,1]
        a_w = self.sigmoid(self.conv_w(y_w))                    # [B,C,1,W]

        att = a_h * a_w                                         # broadcast -> [B,C,H,W]
        return att


class HFFE_Paper(nn.Module):
    """
    A closer reproduction of HAFNet HFFE (Eq.9-16), without your extra modifications.

    Inputs:
      - x_low  : Flow (higher resolution, lower-level)
      - x_high : Fhigh (lower resolution, higher-level), will be upsampled to match x_low

    Output:
      - F_HFFE with out_ch channels (default = c_low), for skip injection usage.
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
    ):
        super().__init__()
        if out_ch is None:
            out_ch = c_low

        self.align_corners = align_corners
        self.sigmoid = nn.Sigmoid()

        # Eq.(9)(10): F' = CBR3(SAM(.))
        self.sam_low = SpatialAttention(kernel_size=sam_kernel)
        self.sam_high = SpatialAttention(kernel_size=sam_kernel)

        self.cbr3_low = ConvNormNonlin(
            conv_op, c_low, c_low, kernel_size=3, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )
        self.cbr3_high = ConvNormNonlin(
            conv_op, c_high, c_high, kernel_size=3, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )

        # Eq.(11)(12): SWM = sigmoid(CBR1(F'))
        # CBR1 in paper = Conv1x1 + BN + ReLU
        self.swm_low = ConvNormNonlin(
            conv_op, c_low, 1, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )
        self.swm_high = ConvNormNonlin(
            conv_op, c_high, 1, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )

        # Eq.(15): SWM_fuse = sigma( CA( Conv1x1([F'low, F'high]) ) )
        self.fuse_pre = ConvNormNonlin(
            conv_op, c_low + c_high, out_ch, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )
        self.coord_att_map = CoordAttMapPaper(
            conv_op, out_ch, reduction=coord_reduction, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )

        # Eq.(16): CBR1( F' + F ) then gate by SWM_fuse
        self.proj_low = ConvNormNonlin(
            conv_op, c_low, out_ch, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )
        self.proj_high = ConvNormNonlin(
            conv_op, c_high, out_ch, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, nonlin_first=nonlin_first
        )

        # Eq.(16) final: Conv1x1( concat(...) )
        self.out_conv = ConvNormNonlin(
            conv_op, out_ch * 2, out_ch, kernel_size=1, padding=0, bias=conv_bias,
            norm_op=None, norm_op_kwargs=None, nonlin=None, nonlin_kwargs=None, nonlin_first=False
        )

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # 1) align resolution: Bi(Fhigh)
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_high_up = F.interpolate(x_high, size=x_low.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        else:
            x_high_up = x_high

        # 2) Eq.(9)(10): F'low, F'high
        f_low_p = self.cbr3_low(self.sam_low(x_low))
        f_high_p = self.cbr3_high(self.sam_high(x_high_up))

        # 3) Eq.(11)(12): SWMlow, SWMhigh (spatial weight matrices)
        swm_low = self.sigmoid(self.swm_low(f_low_p))       # [B,1,H,W]
        swm_high = self.sigmoid(self.swm_high(f_high_p))    # [B,1,H,W]

        # 4) Eq.(13)(14): cross recalibration
        f_low_pp = x_high_up * swm_low
        f_high_pp = x_low * swm_high

        # 5) Eq.(15): SWM_fuse
        fuse_mid = self.fuse_pre(torch.cat([f_low_p, f_high_p], dim=1))
        swm_fuse = self.coord_att_map(fuse_mid)             # [B,out_ch,H,W] in [0,1]

        # 6) Eq.(16): gated projections + final Conv1x1
        low_term = swm_fuse * self.proj_low(f_low_p + x_low)
        high_term = swm_fuse * self.proj_high(f_high_p + x_high_up)

        out = self.out_conv(torch.cat([low_term, high_term], dim=1))
        return out


def build_hffe_pyramid_paper(
    conv_op: Type[_ConvNd],
    features_per_stage: Sequence[int],
    conv_bias: bool,
    norm_op: Optional[Type[nn.Module]],
    norm_op_kwargs: Optional[dict],
    nonlin: Optional[Type[nn.Module]],
    nonlin_kwargs: Optional[dict],
    nonlin_first: bool,
    **hffe_kwargs,
) -> nn.ModuleList:
    """
    Create paper-style HFFE modules for (E0,E1)...(E_{n-2},E_{n-1}).
    out_ch defaults to c_low so it can replace/augment skip features without changing decoder widths.
    """
    hffes = []
    for i in range(len(features_per_stage) - 1):
        c_low = int(features_per_stage[i])
        c_high = int(features_per_stage[i + 1])
        hffes.append(
            HFFE_Paper(
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
                **hffe_kwargs,
            )
        )
    return nn.ModuleList(hffes)
