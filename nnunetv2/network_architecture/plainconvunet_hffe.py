# nnunetv2/network_architecture/plainconvunet_hffe.py
from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.custom_modules.hffe_module import build_hffe_pyramid,ConvNormNonlin


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class PlainConvUNetWithHFFE(PlainConvUNet):
    """
    Non-invasive extension:
    - builds the SAME PlainConvUNet backbone
    - computes HFFE on adjacent encoder skips
    - replaces (or residual-adds) skip features BEFORE decoder
    """
    def __init__(
        self,
        *args,
        hffe_apply_levels: Sequence[int] = (0, 1, 2, 3),  # paper uses i in [1..4] -> 4 modules :contentReference[oaicite:23]{index=23}
        hffe_mode: str = "replace",  # "replace" | "add"
        hffe_enabled: bool = True,
        hffe_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        # PlainConvUNet signature may differ across versions; filter kwargs defensively
        super().__init__(*args, **kwargs)
        self.hffe_enabled = bool(hffe_enabled)
        self.hffe_apply_levels = tuple(int(i) for i in hffe_apply_levels)
        assert hffe_mode in ("replace", "concat")
        self.hffe_mode = hffe_mode
        hffe_kwargs = hffe_kwargs or {}

                # ---- resolve backbone attrs from encoder (your PlainConvUNet stores things there) ----
        enc = getattr(self, "encoder", None)
        if enc is None:
            raise RuntimeError("PlainConvUNetWithHFFE expects PlainConvUNet to have `self.encoder`.")

        # features_per_stage: prefer encoder attribute, otherwise fall back to constructor arg
        fps = getattr(enc, "features_per_stage", None)
        if fps is None:
            # fall back to the passed-in constructor argument (available as local var in __init__)
            # PlainConvUNetWithHFFE.__init__ receives **arch_kwargs which includes features_per_stage
            fps = kwargs.get("features_per_stage", None)
        if fps is None:
            raise RuntimeError(
                "Cannot infer features_per_stage. Expected encoder.features_per_stage "
                "or constructor kwarg `features_per_stage`."
            )

        # conv_op, norm_op, nonlin, etc: prefer encoder attributes, else constructor kwargs
        conv_op = getattr(enc, "conv_op", None) or kwargs.get("conv_op", None)
        if conv_op is None:
            raise RuntimeError("Cannot infer conv_op (expected encoder.conv_op or constructor kwarg conv_op).")

        conv_bias = getattr(enc, "conv_bias", None)
        if conv_bias is None:
            conv_bias = kwargs.get("conv_bias", True)

        norm_op = getattr(enc, "norm_op", None)
        if norm_op is None:
            norm_op = kwargs.get("norm_op", None)

        norm_op_kwargs = getattr(enc, "norm_op_kwargs", None)
        if norm_op_kwargs is None:
            norm_op_kwargs = kwargs.get("norm_op_kwargs", None)

        nonlin = getattr(enc, "nonlin", None)
        if nonlin is None:
            nonlin = kwargs.get("nonlin", None)

        nonlin_kwargs = getattr(enc, "nonlin_kwargs", None)
        if nonlin_kwargs is None:
            nonlin_kwargs = kwargs.get("nonlin_kwargs", None)

        nonlin_first = getattr(enc, "nonlin_first", None)
        if nonlin_first is None:
            nonlin_first = kwargs.get("nonlin_first", False)

        # store (optional, but handy)
        self._hffe_fps = tuple(int(x) for x in fps)
        self._hffe_conv_op = conv_op

        # build HFFE modules for each adjacent pair (E0,E1)...(E_{n-2},E_{n-1})
        self.hffe_modules = build_hffe_pyramid(
            conv_op=conv_op,
            features_per_stage=self._hffe_fps,
            conv_bias=bool(conv_bias),
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            nonlin_first=bool(nonlin_first),
            **hffe_kwargs,
        )
        # adapters for concat mode: Conv1x1(2*c_low -> c_low)
        self.hffe_adapters = nn.ModuleList([
            ConvNormNonlin(
                conv_op=conv_op,
                in_ch=int(self._hffe_fps[i]) * 2,
                out_ch=int(self._hffe_fps[i]),
                kernel_size=1,
                padding=0,
                bias=bool(conv_bias),
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                nonlin_first=bool(nonlin_first),
            )
            for i in range(len(self._hffe_fps) - 1)
        ])


    # -------- knobs for Trainer --------
    def set_hffe_gate_scale(self, scale: float) -> None:
        for m in self.hffe_modules:
            if hasattr(m, "set_gate_scale"):
                m.set_gate_scale(scale)

    def set_hffe_debug(self, flag: bool = True) -> None:
        for m in self.hffe_modules:
            m.debug = bool(flag)

    # -------- forward --------
    def _get_encoder_decoder(self):
        enc = getattr(self, "encoder", None)
        dec = getattr(self, "decoder", None)
        if enc is None or dec is None:
            raise RuntimeError(
                "Cannot find `self.encoder` / `self.decoder` in PlainConvUNet. "
                "Please inspect your dynamic_network_architectures PlainConvUNet implementation."
            )
        return enc, dec

    def forward(self, x: torch.Tensor):
        enc, dec = self._get_encoder_decoder()
        skips = enc(x)

        # some versions may return (skips, bottleneck) or similar; normalize to list[Tensor]
        if isinstance(skips, tuple) and len(skips) >= 1 and isinstance(skips[0], (list, tuple)):
            skips = skips[0]
        if not isinstance(skips, (list, tuple)):
            raise RuntimeError(f"Encoder output type unexpected: {type(skips)}")

        skips = list(skips)
        if self.hffe_enabled:
            max_i = len(skips) - 2
            for i in self.hffe_apply_levels:
                if 0 <= i <= max_i:
                    fused = self.hffe_modules[i](skips[i], skips[i + 1])
                    if self.hffe_mode == "replace":
                        skips[i] = fused
                    else:  # concat
                        skips[i] = self.hffe_adapters[i](torch.cat([skips[i], fused], dim=1))

        # decoder interface differs across versions; try the common one first
        try:
            return dec(skips)
        except TypeError:
            # fallback: (bottleneck, skip_list)
            return dec(skips[-1], skips[:-1])
