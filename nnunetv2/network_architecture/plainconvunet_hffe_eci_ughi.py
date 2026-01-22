# nnunetv2/network_architecture/plainconvunet_hffe_eci_ughi.py
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.custom_modules.hffe_module import build_hffe_pyramid
from nnunetv2.custom_modules.eci_lite import ECILite, ECILiteConfig
from nnunetv2.custom_modules.ughi import UGHI, UGHIConfig


def _bind_plainconv_init_args(args, kwargs) -> Dict[str, Any]:
    sig = inspect.signature(PlainConvUNet.__init__)
    bound = sig.bind_partial(None, *args, **kwargs)
    return dict(bound.arguments)


class PlainConvUNetWithHFFE_ECI_UGHI(PlainConvUNet):
    """
    修正版（显存正确）：
    - 不 wrap decoder.stages（避免二次 cat）
    - 复刻 decoder.forward loop：
        x_up = transpconv(lres_input)
        x_up = UGHI(x_up, HFFE_k, prev_edge_gate)   # 在 cat 之前
        x = cat((x_up, skip_k), 1)
        x = stage(x)
        x = ECI(x)                                  # 在 stage 之后
    - HFFE 只产生 H_k，不改写 skip
    - prev_edge_gate 默认 detach，避免跨 stage 图依赖
    """

    def __init__(
        self,
        *args,
        # HFFE
        hffe_enabled: bool = True,
        hffe_apply_levels: Sequence[int] = (0, 1),
        hffe_kwargs: Optional[dict] = None,
        # ECI
        eci_cfg: Optional[ECILiteConfig] = None,
        eci_apply_levels: Optional[Sequence[int]] = None,          # decoder stage indices
        eci_channels_per_stage: Optional[Sequence[int]] = None,
        # UGHI
        ughi_enabled: bool = True,
        ughi_apply_levels: Optional[Sequence[int]] = None,         # decoder stage indices
        ughi_cfg: Optional[UGHIConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        bound = _bind_plainconv_init_args(args, kwargs)
        enc = getattr(self, "encoder", None)
        dec = getattr(self, "decoder", None)
        if enc is None or dec is None:
            raise RuntimeError("Expected PlainConvUNet to have self.encoder and self.decoder")

        # ---- infer backbone settings ----
        conv_op = getattr(enc, "conv_op", None) or bound.get("conv_op", None) or nn.Conv2d
        conv_bias = getattr(enc, "conv_bias", None)
        if conv_bias is None:
            conv_bias = bound.get("conv_bias", True)

        norm_op = getattr(enc, "norm_op", None) or bound.get("norm_op", None)
        norm_op_kwargs = getattr(enc, "norm_op_kwargs", None) or bound.get("norm_op_kwargs", None)
        nonlin = getattr(enc, "nonlin", None) or bound.get("nonlin", None)
        nonlin_kwargs = getattr(enc, "nonlin_kwargs", None) or bound.get("nonlin_kwargs", None)
        nonlin_first = getattr(enc, "nonlin_first", None)
        if nonlin_first is None:
            nonlin_first = bound.get("nonlin_first", False)

        fps = getattr(enc, "features_per_stage", None) or bound.get("features_per_stage", None)
        if fps is None:
            raise RuntimeError("Cannot infer features_per_stage")
        fps = list(map(int, fps))

        # ---- HFFE modules (adjacent encoder levels) ----
        self.hffe_enabled = bool(hffe_enabled)
        self.hffe_apply_levels = tuple(int(i) for i in hffe_apply_levels)
        hffe_kwargs = hffe_kwargs or {}
        self.hffe_modules = build_hffe_pyramid(
            conv_op=conv_op,
            features_per_stage=tuple(fps),
            conv_bias=bool(conv_bias),
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            nonlin_first=bool(nonlin_first),
            **hffe_kwargs,
        )

        # ---- decoder stage count ----
        if not hasattr(dec, "stages") or not hasattr(dec, "transpconvs") or not hasattr(dec, "seg_layers"):
            raise RuntimeError("Your decoder must have attributes: stages, transpconvs, seg_layers")

        self._n_dec = len(dec.stages)

        # ---- infer channels per decoder stage ----
        if eci_channels_per_stage is None:
            guess = list(reversed(fps[:-1]))
            if len(guess) != self._n_dec:
                raise RuntimeError("channels_per_stage inference mismatch. Pass eci_channels_per_stage.")
            channels_per_stage = guess
        else:
            channels_per_stage = [int(c) for c in eci_channels_per_stage]
            if len(channels_per_stage) != self._n_dec:
                raise ValueError("eci_channels_per_stage length must match decoder stages")

        # apply sets
        eci_apply = set(range(self._n_dec)) if eci_apply_levels is None else set(int(i) for i in eci_apply_levels)
        ughi_apply = set(range(self._n_dec)) if ughi_apply_levels is None else set(int(i) for i in ughi_apply_levels)

        self.eci_cfg = eci_cfg if eci_cfg is not None else ECILiteConfig()
        self.ughi_cfg = ughi_cfg if ughi_cfg is not None else UGHIConfig()

        # ---- build per-stage ECI/UGHI lists ----
        self.eci_modules = nn.ModuleList()
        self.ughi_modules = nn.ModuleList()

        for s in range(self._n_dec):
            ch = int(channels_per_stage[s])

            # stage s corresponds to encoder level k = (n_enc - 2) - s
            # H_k has channels fps[k]
            k = (len(fps) - 2) - s
            hffe_ch = int(fps[k]) if 0 <= k < (len(fps) - 1) else ch

            if bool(ughi_enabled) and (s in ughi_apply):
                self.ughi_modules.append(
                    UGHI(conv_op=conv_op, up_ch=ch, hffe_ch=hffe_ch, cfg=self.ughi_cfg, conv_bias=bool(conv_bias))
                )
            else:
                self.ughi_modules.append(nn.Identity())

            if s in eci_apply:
                self.eci_modules.append(
                    ECILite(
                        conv_op=conv_op,
                        channels=ch,
                        cfg=self.eci_cfg,
                        conv_bias=bool(conv_bias),
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                )
            else:
                self.eci_modules.append(nn.Identity())

        # runtime cache for trainer edge loss
        self._edge_logits_cache: List[Optional[torch.Tensor]] = [None] * self._n_dec

    # ---- helpers for Trainer ----
    def reset_edge_cache(self):
        for i in range(self._n_dec):
            self._edge_logits_cache[i] = None

    def get_eci_edge_logits(self) -> List[Optional[torch.Tensor]]:
        return self._edge_logits_cache

    def set_eci_inject_scale(self, s: float) -> None:
        for m in self.eci_modules:
            if hasattr(m, "set_inject_scale"):
                m.set_inject_scale(float(s))

    def forward(self, x: torch.Tensor):
        self.reset_edge_cache()

        enc = self.encoder
        dec = self.decoder

        skips = enc(x)
        if isinstance(skips, tuple) and len(skips) >= 1 and isinstance(skips[0], (list, tuple)):
            skips = skips[0]
        skips = list(skips)

        # ---- compute pure HFFE features H_i (no overwrite) ----
        hffe_feats: List[Optional[torch.Tensor]] = [None] * (len(skips) - 1)
        if self.hffe_enabled:
            max_i = len(skips) - 2
            for i in self.hffe_apply_levels:
                if 0 <= int(i) <= max_i:
                    hffe_feats[int(i)] = self.hffe_modules[int(i)](skips[int(i)], skips[int(i) + 1])

        # ---- custom decoder loop (no extra cat) ----
        lres_input = skips[-1]
        seg_outputs = []
        prev_gate: Optional[torch.Tensor] = None  # (B,1,H,W) prob map, detached recommended

        # nnUNet toggles deep supervision during inference; different code paths use different flags.
        # We resolve do_ds in a robust order so that predictor can disable it cleanly.
        do_ds = getattr(self, "do_ds", None)
        if do_ds is None:
            do_ds = getattr(self, "deep_supervision", None)
        if do_ds is None:
            do_ds = getattr(self, "enable_deep_supervision", None)
        if do_ds is None:
            do_ds = getattr(dec, "deep_supervision", False)
        do_ds = bool(do_ds)

        for s in range(len(dec.stages)):
            x_up = dec.transpconvs[s](lres_input)

            # map stage s -> encoder level k
            k = (len(skips) - 2) - s
            h_k = hffe_feats[k] if 0 <= k < len(hffe_feats) else None

            # UGHI: only on vertical branch, BEFORE cat
            ughi = self.ughi_modules[s]
            if not isinstance(ughi, nn.Identity):
                x_up = ughi(x_up, h_k, prev_gate)

            # original skip (pure)
            x_cat = torch.cat((x_up, skips[-(s + 2)]), 1)

            # conv stage
            x_stage = dec.stages[s](x_cat)

            # ECI: post-stage
            eci = self.eci_modules[s]
            if not isinstance(eci, nn.Identity):
                x_stage, edge_logits = eci(x_stage, return_edge_logits=True)
                self._edge_logits_cache[s] = edge_logits

                # gate for next stage: prob map, detach to avoid cross-stage backward graph
                prev_gate = torch.sigmoid(edge_logits)
                if getattr(self.ughi_cfg, "detach_prev_edge_gate", True):
                    prev_gate = prev_gate.detach()
            else:
                self._edge_logits_cache[s] = None
                prev_gate = None

            # deep supervision behavior must match nnUNet expectations:
            # - during training/regular validation: return list (deep supervision)
            # - during inference/sliding-window: return a Tensor (no deep supervision)
            if do_ds:
                seg_outputs.append(dec.seg_layers[s](x_stage))
            elif s == (len(dec.stages) - 1):
                out = dec.seg_layers[-1](x_stage)

            lres_input = x_stage

        if do_ds:
            seg_outputs = seg_outputs[::-1]
            return seg_outputs
        return out
