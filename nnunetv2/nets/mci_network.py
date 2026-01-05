"""
2D-only MCI adaptation for nnUNetV2.

Key design goals:
- Keep nnUNet encoder unchanged.
- Replace decoder with an MCI-augmented decoder (Seg + Edge + Centerline cues).
- Make inference compatible with nnUNet predictors: forward() returns ONLY segmentation logits
  unless return_aux=True is passed (used by the trainer).
"""

from __future__ import annotations

from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp


Tensor = torch.Tensor


class MCIBlock2D(nn.Module):
    """
    Paper-aligned MCI block adapted to auxiliary tasks Edge + CenterLine (CL).

    Branches:
      - Seg feature branch
      - Edge feature branch
      - CL feature branch

    Aggregation:
      - M2CFS: Seg -> (Edge, CL)
      - C2MFA: (Edge, CL) -> Seg  (with self-attn enhancement for aux branches)
    """
    def __init__(self, channels: int, *, norm_op=nn.InstanceNorm2d):
        super().__init__()
        # task-specific feature projections
        self.conv_seg = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv_edge = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv_cl = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # IMPORTANT: each branch gets its own norm to avoid unintended task coupling
        self.norm_seg = norm_op(channels, affine=True)
        self.norm_edge = norm_op(channels, affine=True)
        self.norm_cl = norm_op(channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        # attention convs (Conv + sigmoid)
        self.attn_seg = nn.Conv2d(channels, channels, 1, bias=True)
        self.attn_edge = nn.Conv2d(channels, channels, 1, bias=True)
        self.attn_cl = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 1) branch features
        f_seg = self.act(self.norm_seg(self.conv_seg(x)))
        f_edge = self.act(self.norm_edge(self.conv_edge(x)))
        f_cl = self.act(self.norm_cl(self.conv_cl(x)))

        # 2) M2CFS: Seg -> Edge/CL supplementation in weak-response areas
        w_edge = 1.0 - torch.sigmoid(self.attn_edge(f_edge))
        f_edge2 = f_edge + w_edge * f_seg

        w_cl = 1.0 - torch.sigmoid(self.attn_cl(f_cl))
        f_cl2 = f_cl + w_cl * f_seg

        # 3) C2MFA: aux self-attn enhancement + aux -> Seg aggregation
        f_edge_attn = torch.sigmoid(self.attn_edge(f_edge)) * f_edge
        f_cl_attn = torch.sigmoid(self.attn_cl(f_cl)) * f_cl

        w_seg = 1.0 - torch.sigmoid(self.attn_seg(f_seg))
        f_seg2 = f_seg + w_seg * (f_edge_attn + f_cl_attn)

        return f_seg2, f_edge2, f_cl2


class MCIDecoder2D(nn.Module):
    """
    A 2D nnUNet-style decoder augmented with MCI blocks at each decoder stage.

    Returns:
      - If return_aux=False (default): segmentation logits (Tensor or List[Tensor] if deep_supervision)
      - If return_aux=True: (seg_logits_list, edge_logits_list, cl_logits_list) all ordered high->low.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        deep_supervision: bool,
        conv_per_stage: int = 2,
        nonlin_first: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        # Expect encoder to expose these (dynamic_network_architectures encoders do)
        if not hasattr(encoder, "output_channels") or not hasattr(encoder, "strides"):
            raise RuntimeError(
                "Encoder does not expose output_channels/strides. "
                "This MCI decoder expects a dynamic_network_architectures encoder."
            )

        enc_ch = list(encoder.output_channels)  # low->high depth
        strides = list(encoder.strides)

        # decoder stages from bottleneck up to highest resolution
        self.transp_convs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.mci_blocks = nn.ModuleList()

        self.seg_heads = nn.ModuleList()
        self.edge_heads = nn.ModuleList()
        # CL: per-class foreground heatmaps (C-1); if binary, this equals 1
        self.cl_channels = max(1, self.num_classes - 1)
        self.cl_heads = nn.ModuleList()

        # create stages (iterate from deepest to shallow)
        for i in range(len(enc_ch) - 1, 0, -1):
            in_ch = enc_ch[i]
            skip_ch = enc_ch[i - 1]
            out_ch = skip_ch

            # upsample convtranspose matching encoder conv op
            TranspConv = get_matching_convtransp(encoder.conv_op)
            ks = tuple(int(s) for s in strides[i])
            transp = TranspConv(in_ch, out_ch, kernel_size=ks, stride=ks, bias=False)
            self.transp_convs.append(transp)

            # after concat: channels = out_ch + skip_ch
            stage_in_ch = out_ch + skip_ch
            # keep nnUNet decoder capacity: stacked conv blocks
            stage = StackedConvBlocks(
                num_convs=conv_per_stage,
                conv_op=encoder.conv_op,
                input_channels=stage_in_ch,
                output_channels=out_ch,
                kernel_size=encoder.kernel_sizes[i - 1],
                initial_stride=1,
                conv_bias=True,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
                nonlin_first=nonlin_first,
            )
            self.stages.append(stage)

            # plugin MCI block
            self.mci_blocks.append(MCIBlock2D(out_ch, norm_op=encoder.norm_op))

            # heads at this resolution
            self.seg_heads.append(nn.Conv2d(out_ch, self.num_classes, 1, bias=True))
            self.edge_heads.append(nn.Conv2d(out_ch, 1, 1, bias=True))
            self.cl_heads.append(nn.Conv2d(out_ch, self.cl_channels, 1, bias=True))

        # note: lists are deepest->shallowest in construction order; forward will reverse to high->low

    def forward(self, x: Tensor, *, return_aux: bool = False):
        skips = self.encoder(x)

        if not isinstance(skips, (list, tuple)) or len(skips) < 2:
            raise RuntimeError(
                "Encoder forward must return a list/tuple of skip feature maps (including bottleneck)."
            )

        res = skips[-1]  # bottleneck
        seg_outs: List[Tensor] = []
        edge_outs: List[Tensor] = []
        cl_outs: List[Tensor] = []

        # decode: from bottleneck to high-res
        for i in range(len(self.stages)):
            skip = skips[-(i + 2)]
            res = self.transp_convs[i](res)
            res = torch.cat((res, skip), dim=1)
            res = self.stages[i](res)

            f_seg, f_edge, f_cl = self.mci_blocks[i](res)
            res = f_seg  # continue with refined seg feature

            seg_outs.append(self.seg_heads[i](f_seg))
            edge_outs.append(self.edge_heads[i](f_edge))
            cl_outs.append(self.cl_heads[i](f_cl))

        # reverse to high->low to match nnUNet DS target ordering
        seg_outs = seg_outs[::-1]
        edge_outs = edge_outs[::-1]
        cl_outs = cl_outs[::-1]

        if return_aux:
            return seg_outs, edge_outs, cl_outs

        # inference-compatible return: only segmentation logits
        if self.deep_supervision:
            return seg_outs
        return seg_outs[0]


class MCIUNetWrapper2D(nn.Module):
    """
    Wrap an nnUNet network (built by nnUNetTrainer) by reusing its encoder and replacing decoder.

    This is intentionally minimal to avoid coupling to nnUNet internals.
    """
    def __init__(self, base_net: nn.Module, num_classes: int, *, deep_supervision: bool):
        super().__init__()
        if not hasattr(base_net, "encoder"):
            raise RuntimeError("base_net must expose .encoder to be wrapped with MCIUNetWrapper2D")
        self.encoder = base_net.encoder
        self.decoder = MCIDecoder2D(self.encoder, num_classes, deep_supervision=deep_supervision)

    def forward(self, x: Tensor, *, return_aux: bool = False):
        # decoder calls encoder internally; keep signature compatible
        return self.decoder(x, return_aux=return_aux)
