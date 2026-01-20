# nnunetv2/network_architecture/plainconvunet_hffe_eci.py
from __future__ import annotations

"""
PlainConvUNetWithHFFEAndECI (fixed baseline)
- HFFE: edits encoder skips before decoder
- ECI: wraps decoder stages (post-stage) with edge-conditioned interaction
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from nnunetv2.network_architecture.plainconvunet_hffe import PlainConvUNetWithHFFE
from nnunetv2.custom_modules.eci_lite import ECILiteConfig
from nnunetv2.network_architecture.plainconvunet_eci import _locate_decoder_stages, _DecoderStageWrapper
from nnunetv2.custom_modules.eci_lite import ECILite


class PlainConvUNetWithHFFEAndECI(PlainConvUNetWithHFFE):
    def __init__(
        self,
        *args,
        # HFFE args
        hffe_apply_levels: Sequence[int] = (0, 1, 2, 3),
        hffe_mode: str = "replace",
        hffe_enabled: bool = True,
        hffe_kwargs: Optional[dict] = None,
        # ECI args
        eci_cfg: Optional[ECILiteConfig] = None,
        eci_apply_levels: Optional[Sequence[int]] = None,
        eci_channels_per_stage: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            hffe_apply_levels=hffe_apply_levels,
            hffe_mode=hffe_mode,
            hffe_enabled=hffe_enabled,
            hffe_kwargs=hffe_kwargs,
            **kwargs,
        )

        enc = getattr(self, "encoder", None)
        conv_op = getattr(enc, "conv_op", None) or kwargs.get("conv_op", None) or nn.Conv2d
        conv_bias = getattr(enc, "conv_bias", None)
        if conv_bias is None:
            conv_bias = kwargs.get("conv_bias", True)

        norm_op = getattr(enc, "norm_op", None) or kwargs.get("norm_op", None)
        norm_op_kwargs = getattr(enc, "norm_op_kwargs", None) or kwargs.get("norm_op_kwargs", None)
        nonlin = getattr(enc, "nonlin", None) or kwargs.get("nonlin", None)
        nonlin_kwargs = getattr(enc, "nonlin_kwargs", None) or kwargs.get("nonlin_kwargs", None)

        self.eci_cfg = eci_cfg if eci_cfg is not None else ECILiteConfig()

        stage_attr, stages = _locate_decoder_stages(self.decoder)
        n_dec = len(stages)

        if eci_channels_per_stage is None:
            fps = getattr(enc, "features_per_stage", None) or kwargs.get("features_per_stage", None)
            if fps is None:
                raise RuntimeError("Cannot infer features_per_stage; pass eci_channels_per_stage explicitly.")
            fps = list(fps)
            guess = list(reversed(fps[:-1]))
            if len(guess) != n_dec:
                raise RuntimeError(
                    f"Decoder stages={n_dec} but inferred channels={len(guess)} from features_per_stage. "
                    f"Pass eci_channels_per_stage explicitly."
                )
            channels_per_stage = guess
        else:
            channels_per_stage = [int(c) for c in eci_channels_per_stage]
            if len(channels_per_stage) != n_dec:
                raise ValueError(f"eci_channels_per_stage length must match decoder stages ({n_dec}).")

        if eci_apply_levels is None:
            apply = set(range(n_dec))
        else:
            apply = set(int(i) for i in eci_apply_levels if 0 <= int(i) < n_dec)

        self._eci_edge_logits = [None] * n_dec

        wrapped = []
        for i, stage in enumerate(stages):
            if i in apply:
                eci = ECILite(
                    conv_op=conv_op,
                    channels=channels_per_stage[i],
                    cfg=self.eci_cfg,
                    conv_bias=bool(conv_bias),
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            else:
                eci = None
            wrapped.append(_DecoderStageWrapper(stage, eci, self, i))

        setattr(self.decoder, stage_attr, nn.ModuleList(wrapped))

    def reset_eci_cache(self) -> None:
        for i in range(len(self._eci_edge_logits)):
            self._eci_edge_logits[i] = None

    def get_eci_edge_logits(self):
        return self._eci_edge_logits

    def set_eci_inject_scale(self, s: float) -> None:
        stage_attr, stages = _locate_decoder_stages(self.decoder)
        for st in stages:
            eci = getattr(st, "eci", None)
            if eci is not None:
                eci.set_inject_scale(float(s))

    def forward(self, x: torch.Tensor):
        self.reset_eci_cache()
        return super().forward(x)
