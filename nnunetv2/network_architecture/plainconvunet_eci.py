# nnunetv2/network_architecture/plainconvunet_eci.py
# SPDX-License-Identifier: MIT
"""
PlainConvUNetWithECI:
- Extends the vanilla PlainConvUNet (dynamic_network_architectures) with ECI-Lite blocks
  inserted *inside decoder stages* (post-stage conv stack).

Key assumptions (matches nnUNet v2 PlainConvUNet in MIC-DKFZ/nnUNet):
- self.encoder and self.decoder exist.
- self.decoder has a ModuleList of per-resolution conv stacks under one of:
    stages / blocks / decoder_stages
  and their order is from low-res (closest bottleneck) -> high-res (full resolution).
If your local nnUNet version differs, adjust `_locate_decoder_stages`.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    # nnUNet v2 uses dynamic_network_architectures for PlainConvUNet
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except Exception as e:
    raise ImportError(
        "Could not import PlainConvUNet from dynamic_network_architectures. "
        "Make sure nnUNet v2 and dynamic_network_architectures are installed."
    ) from e

# Your custom module (place this file under nnunetv2/custom_modules/)
from nnunetv2.custom_modules.eci_lite import ECILiteConfig, build_eci_lite_pyramid


def _bind_plainconv_init_args(args, kwargs) -> Dict[str, Any]:
    """
    Bind PlainConvUNet.__init__ signature to (args, kwargs) so we can reliably read
    features_per_stage, conv_op, etc even if caller used positional args.
    """
    sig = inspect.signature(PlainConvUNet.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)  # 'self' placeholder
    return dict(bound.arguments)


def _locate_decoder_stages(decoder: nn.Module) -> Tuple[str, nn.ModuleList]:
    """
    Return (attr_name, module_list) for decoder stages.

    We intentionally support multiple attr names to reduce version coupling.
    """
    candidates = ("stages", "blocks", "decoder_stages", "stages_decoder", "conv_blocks")
    for name in candidates:
        v = getattr(decoder, name, None)
        if isinstance(v, nn.ModuleList) and len(v) > 0:
            return name, v
    raise RuntimeError(
        f"Cannot locate decoder stages ModuleList. Tried attrs={candidates}. "
        f"Available attrs={sorted([k for k in dir(decoder) if not k.startswith('_')])}"
    )


class _DecoderStageWithECI(nn.Module):
    """
    Wrap a per-stage conv stack so that:
      y = stage(x)
      y = eci(y)
    and store edge logits for this stage (optional).
    """
    def __init__(self, stage: nn.Module, eci: nn.Module, edge_cache: list, stage_idx: int):
        super().__init__()
        self.stage = stage
        self.eci = eci
        # IMPORTANT: do NOT hold a reference to the parent nn.Module here.
        # Storing the full network as a submodule creates a module-cycle and breaks
        # nn.Module.apply()/state_dict() traversal. Use a plain python list as cache.
        self.edge_cache = edge_cache
        self.stage_idx = int(stage_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stage(x)
        # Store edge logits only when edge head is enabled.
        # When edge loss is disabled, trainer may disable edge_head to save compute/memory.
        need_edge = bool(getattr(self.eci, 'edge_head_enabled', True))
        out = self.eci(y, return_edge_logits=need_edge)
        if need_edge:
            y2, edge_logits = out
        else:
            # Some implementations may return only y, or (y, None). Handle both.
            if isinstance(out, (tuple, list)):
                y2 = out[0]
            else:
                y2 = out
            edge_logits = None
        self.edge_cache[self.stage_idx] = edge_logits
        return y2


class PlainConvUNetWithECI(PlainConvUNet):
    """
    Vanilla PlainConvUNet + ECI-Lite after each decoder stage.

    Public API:
      - get_eci_edge_logits(): List[Tensor|None]
      - set_eci_inject_scale(s): sets cfg.inject_scale for all stages
      - set_eci_edge_head_enabled(enabled): toggles edge head for all stages
    """
    def __init__(
        self,
        *args,
        eci_cfg: Optional[ECILiteConfig] = None,
        eci_apply_levels: Optional[Sequence[int]] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        bound = _bind_plainconv_init_args(args, kwargs)
        features_per_stage = bound.get("features_per_stage", None)
        conv_op = bound.get("conv_op", None)
        conv_bias = bool(bound.get("conv_bias", True))
        norm_op = bound.get("norm_op", None)
        norm_op_kwargs = bound.get("norm_op_kwargs", None)
        nonlin = bound.get("nonlin", None)
        nonlin_kwargs = bound.get("nonlin_kwargs", None)

        if features_per_stage is None or conv_op is None:
            raise RuntimeError(
                "Could not infer required init args (features_per_stage, conv_op). "
                "Check PlainConvUNet.__init__ signature in your nnUNet version."
            )

        self.eci_cfg = eci_cfg if eci_cfg is not None else ECILiteConfig()
        n_dec = len(features_per_stage) - 1

        if eci_apply_levels is None:
            levels = list(range(n_dec))          # all
        else:
            levels = list(eci_apply_levels)      # allow []

        norm_levels = []
        for i in levels:
            i = int(i)
            if i < 0:
                i = n_dec + i                    # -1 -> last
            if 0 <= i < n_dec:
                norm_levels.append(i)

        # IMPORTANT: distinguish between "user explicitly gave []" (means NONE)
        # and "user gave something but all indices invalid" (fallback to last)
        if eci_apply_levels is not None and len(levels) > 0 and len(norm_levels) == 0:
            norm_levels = [n_dec - 1]

        self.eci_apply_levels = set(norm_levels)


        # Build ECI modules for decoder stages
        self.eci_modules = build_eci_lite_pyramid(
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            cfg=self.eci_cfg,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        # Cache for edge logits (updated each forward)
        self._eci_edge_logits: List[Optional[torch.Tensor]] = [None] * len(self.eci_modules)

        # Inject by wrapping decoder stages
        stage_attr, stages = _locate_decoder_stages(self.decoder)
        if len(stages) != len(self.eci_modules):
            raise RuntimeError(
                f"Decoder stages length mismatch: decoder.{stage_attr} has {len(stages)} stages "
                f"but expected {len(self.eci_modules)} (n_stages-1). "
                "If your nnUNet version differs, adjust channel mapping and stage discovery."
            )

        wrapped = nn.ModuleList()
        for i, stage in enumerate(stages):
            if i in self.eci_apply_levels:
                wrapped.append(_DecoderStageWithECI(stage, self.eci_modules[i], edge_cache=self._eci_edge_logits, stage_idx=i))
            else:
                wrapped.append(stage)
        setattr(self.decoder, stage_attr, wrapped)

    def reset_eci_cache(self) -> None:
        # IMPORTANT: in-place reset (do NOT rebind the list), because decoder wrappers
        # hold a reference to this list as edge_cache.
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

    def forward(self, x: torch.Tensor):
        # Ensure edge cache is cleared each call
        self.reset_eci_cache()
        # Use the original encoder+decoder path. Decoder stages are already wrapped.
        return super().forward(x)
