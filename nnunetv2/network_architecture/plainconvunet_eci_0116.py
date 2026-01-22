# nnunetv2/network_architecture/plainconvunet_eci.py
from __future__ import annotations

"""
PlainConvUNetWithECI
- Wraps PlainConvUNet decoder stages and applies ECI-Lite AFTER each decoder stage (post-stage conv stack).

Critical engineering note:
- DO NOT store a reference to the parent network (nn.Module) inside child modules.
  That creates a cyclic module graph (parent becomes a child), and nn.Module.apply()/modules()
  will recurse forever. We cache edge_logits inside each wrapper instead.
"""

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from code.nnUNet.nnunetv2.custom_modules.eci_lite_0116 import ECILite, ECILiteConfig


def _bind_plainconv_init_args(args, kwargs) -> Dict[str, Any]:
    sig = inspect.signature(PlainConvUNet.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)
    return dict(bound.arguments)


def _locate_decoder_stages(decoder: nn.Module) -> Tuple[str, nn.ModuleList]:
    candidates = ("stages", "blocks", "decoder_stages", "stages_decoder", "conv_blocks")
    for name in candidates:
        v = getattr(decoder, name, None)
        if isinstance(v, nn.ModuleList) and len(v) > 0:
            return name, v
    raise RuntimeError(
        f"Cannot locate decoder stages ModuleList. Tried attrs={candidates}. "
        f"Available attrs={sorted([k for k in dir(decoder) if not k.startswith('_')])}"
    )


class _DecoderStageWrapper(nn.Module):
    """
    Wrap one decoder stage:
      y = stage(x)
      y = ECI(y) (optional)
    Cache edge_logits in self.edge_logits_cache (Tensor|None).
    """
    def __init__(self, stage: nn.Module, eci: Optional[ECILite], stage_idx: int):
        super().__init__()
        self.stage = stage
        self.eci = eci
        self.stage_idx = int(stage_idx)
        self.edge_logits_cache: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self.edge_logits_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stage(x)
        if self.eci is None:
            self.edge_logits_cache = None
            return y
        y2, edge_logits = self.eci(y, return_edge_logits=True)
        self.edge_logits_cache = edge_logits
        return y2


class PlainConvUNetWithECI(PlainConvUNet):
    """
    Vanilla PlainConvUNet + ECI-Lite after selected decoder stages.

    Public helpers:
      - get_eci_edge_logits(): List[Tensor|None] aligned with decoder stage order
      - set_eci_inject_scale(s): set scale for all active ECI modules
    """
    def __init__(
        self,
        *args,
        eci_cfg: Optional[ECILiteConfig] = None,
        eci_apply_levels: Optional[Sequence[int]] = None,
        eci_channels_per_stage: Optional[Sequence[int]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        bound = _bind_plainconv_init_args(args, kwargs)
        enc = getattr(self, "encoder", None)

        conv_op = getattr(enc, "conv_op", None) or bound.get("conv_op", None) or nn.Conv2d
        conv_bias = getattr(enc, "conv_bias", None)
        if conv_bias is None:
            conv_bias = bound.get("conv_bias", True)

        norm_op = getattr(enc, "norm_op", None) or bound.get("norm_op", None)
        norm_op_kwargs = getattr(enc, "norm_op_kwargs", None) or bound.get("norm_op_kwargs", None)
        nonlin = getattr(enc, "nonlin", None) or bound.get("nonlin", None)
        nonlin_kwargs = getattr(enc, "nonlin_kwargs", None) or bound.get("nonlin_kwargs", None)

        self.eci_cfg = eci_cfg if eci_cfg is not None else ECILiteConfig()

        stage_attr, stages = _locate_decoder_stages(self.decoder)
        n_dec = len(stages)

        # infer channels per decoder stage
        if eci_channels_per_stage is None:
            fps = getattr(enc, "features_per_stage", None) or bound.get("features_per_stage", None)
            if fps is None:
                raise RuntimeError("Cannot infer features_per_stage; pass eci_channels_per_stage explicitly.")
            fps = list(fps)
            guess = list(reversed(fps[:-1]))  # typical mapping
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

        # which stages to apply ECI
        if eci_apply_levels is None:
            apply = set(range(n_dec))
        else:
            apply = set(int(i) for i in eci_apply_levels if 0 <= int(i) < n_dec)

        wrapped: List[_DecoderStageWrapper] = []
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
            wrapped.append(_DecoderStageWrapper(stage, eci, i))

        setattr(self.decoder, stage_attr, nn.ModuleList(wrapped))

    # ---- cache & scale helpers ----
    def _wrapped_stages(self) -> List[_DecoderStageWrapper]:
        _, stages = _locate_decoder_stages(self.decoder)
        return list(stages)  # type: ignore[return-value]

    def reset_eci_cache(self) -> None:
        for st in self._wrapped_stages():
            st.reset_cache()

    def get_eci_edge_logits(self) -> List[Optional[torch.Tensor]]:
        return [st.edge_logits_cache for st in self._wrapped_stages()]

    def set_eci_inject_scale(self, s: float) -> None:
        for st in self._wrapped_stages():
            if st.eci is not None:
                st.eci.set_inject_scale(float(s))

    def forward(self, x: torch.Tensor):
        self.reset_eci_cache()
        return super().forward(x)
