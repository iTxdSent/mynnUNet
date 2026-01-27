# nnunetv2/network_architecture/plainconvunet_hffe_eci.py
from __future__ import annotations
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.custom_modules.hffe_module import build_hffe_pyramid, ConvNormNonlin
from nnunetv2.custom_modules.eci_lite import ECILiteConfig, build_eci_lite_pyramid


def _locate_decoder_stages(decoder: nn.Module) -> Tuple[str, nn.ModuleList]:
    candidates = ("stages", "blocks", "decoder_stages", "stages_decoder", "conv_blocks")
    for name in candidates:
        v = getattr(decoder, name, None)
        if isinstance(v, nn.ModuleList) and len(v) > 0:
            return name, v
    raise RuntimeError(f"Cannot locate decoder stages ModuleList. Tried {candidates}.")


class _DecoderStageWithECI(nn.Module):
    def __init__(self, stage: nn.Module, eci: nn.Module, edge_cache: list, stage_idx: int):
        super().__init__()
        self.stage = stage
        self.eci = eci
        self.edge_cache = edge_cache
        self.stage_idx = int(stage_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stage(x)
        need_edge = bool(getattr(self.eci, "edge_head_enabled", True))
        out = self.eci(y, return_edge_logits=need_edge)
        if need_edge:
            y2, edge_logits = out
        else:
            y2 = out[0] if isinstance(out, (tuple, list)) else out
            edge_logits = None
        self.edge_cache[self.stage_idx] = edge_logits
        return y2


class PlainConvUNetWithHFFE_ECI(PlainConvUNet):
    def __init__(
        self,
        *args,
        # ---- HFFE knobs ----
        hffe_enabled: bool = True,
        hffe_apply_levels: Sequence[int] = (0, 1),
        hffe_mode: str = "concat",  # "replace"|"concat"
        hffe_kwargs: Optional[dict] = None,
        # ---- ECI knobs ----
        eci_cfg: Optional[ECILiteConfig] = None,
        eci_apply_levels: Optional[Sequence[int]] = (-2, -1),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert hffe_mode in ("replace", "concat")
        self.hffe_enabled = bool(hffe_enabled)
        self.hffe_apply_levels = tuple(int(i) for i in hffe_apply_levels)
        self.hffe_mode = hffe_mode
        hffe_kwargs = hffe_kwargs or {}

        # ---- infer backbone meta (same pattern as your HFFE net) ----
        enc = getattr(self, "encoder", None)
        if enc is None:
            raise RuntimeError("Expected PlainConvUNet to have self.encoder")

        fps = getattr(enc, "features_per_stage", None) or kwargs.get("features_per_stage", None)
        if fps is None:
            raise RuntimeError("Cannot infer features_per_stage")
        self._fps = tuple(int(x) for x in fps)

        conv_op = getattr(enc, "conv_op", None) or kwargs.get("conv_op", None)
        conv_bias = getattr(enc, "conv_bias", None)
        if conv_bias is None:
            conv_bias = kwargs.get("conv_bias", True)

        norm_op = getattr(enc, "norm_op", None) or kwargs.get("norm_op", None)
        norm_op_kwargs = getattr(enc, "norm_op_kwargs", None) or kwargs.get("norm_op_kwargs", None)
        nonlin = getattr(enc, "nonlin", None) or kwargs.get("nonlin", None)
        nonlin_kwargs = getattr(enc, "nonlin_kwargs", None) or kwargs.get("nonlin_kwargs", None)
        nonlin_first = getattr(enc, "nonlin_first", None)
        if nonlin_first is None:
            nonlin_first = kwargs.get("nonlin_first", False)

        # ---- build HFFE pyramid + adapters ----
        self.hffe_modules = build_hffe_pyramid(
            conv_op=conv_op,
            features_per_stage=self._fps,
            conv_bias=bool(conv_bias),
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            nonlin_first=bool(nonlin_first),
            **hffe_kwargs,
        )
        self.hffe_adapters = nn.ModuleList([
            ConvNormNonlin(
                conv_op=conv_op,
                in_ch=int(self._fps[i]) * 2,
                out_ch=int(self._fps[i]),
                kernel_size=1,
                padding=0,
                bias=bool(conv_bias),
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                nonlin_first=bool(nonlin_first),
            )
            for i in range(len(self._fps) - 1)
        ])

        # ---- build ECI pyramid ----
        self.eci_cfg = eci_cfg if eci_cfg is not None else ECILiteConfig()
        self.eci_modules = build_eci_lite_pyramid(
            features_per_stage=self._fps,
            conv_op=conv_op,
            cfg=self.eci_cfg,
            conv_bias=bool(conv_bias),
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self._eci_edge_logits: List[Optional[torch.Tensor]] = [None] * len(self.eci_modules)

        # normalize eci_apply_levels (decoder order: low->high, allow negative)
        n_dec = len(self._fps) - 1
        if eci_apply_levels is None:
            levels = list(range(n_dec))
        else:
            levels = []
            for i in eci_apply_levels:
                ii = int(i)
                if ii < 0:
                    ii = n_dec + ii
                if 0 <= ii < n_dec:
                    levels.append(ii)
        self.eci_apply_levels = set(levels)

        stage_attr, stages = _locate_decoder_stages(self.decoder)
        if len(stages) != len(self.eci_modules):
            raise RuntimeError("Decoder stage count mismatch with ECI modules")

        wrapped = nn.ModuleList()
        for i, stage in enumerate(stages):
            if i in self.eci_apply_levels:
                wrapped.append(_DecoderStageWithECI(stage, self.eci_modules[i], self._eci_edge_logits, i))
            else:
                wrapped.append(stage)
        setattr(self.decoder, stage_attr, wrapped)

    # ---- HFFE knobs ----
    def set_hffe_gate_scale(self, scale: float) -> None:
        for m in self.hffe_modules:
            if hasattr(m, "set_gate_scale"):
                m.set_gate_scale(scale)

    def set_hffe_debug(self, flag: bool = True) -> None:
        for m in self.hffe_modules:
            m.debug = bool(flag)

    # ---- ECI knobs ----
    def reset_eci_cache(self) -> None:
        for i in range(len(self._eci_edge_logits)):
            self._eci_edge_logits[i] = None

    def get_eci_edge_logits(self) -> List[Optional[torch.Tensor]]:
        return self._eci_edge_logits

    def set_eci_inject_scale(self, scale: float) -> None:
        for m in self.eci_modules:
            m.set_inject_scale(float(scale))

    def set_eci_edge_head_enabled(self, enabled: bool) -> None:
        for m in self.eci_modules:
            m.set_edge_head_enabled(bool(enabled))

    # ---- forward ----
    def forward(self, x: torch.Tensor):
        self.reset_eci_cache()

        skips = self.encoder(x)
        if isinstance(skips, tuple) and len(skips) >= 1 and isinstance(skips[0], (list, tuple)):
            skips = skips[0]
        skips = list(skips)

        if self.hffe_enabled:
            max_i = len(skips) - 2
            for i in self.hffe_apply_levels:
                if 0 <= i <= max_i:
                    fused = self.hffe_modules[i](skips[i], skips[i + 1])
                    if self.hffe_mode == "replace":
                        skips[i] = fused
                    else:
                        skips[i] = self.hffe_adapters[i](torch.cat([skips[i], fused], dim=1))

        try:
            return self.decoder(skips)
        except TypeError:
            return self.decoder(skips[-1], skips[:-1])
