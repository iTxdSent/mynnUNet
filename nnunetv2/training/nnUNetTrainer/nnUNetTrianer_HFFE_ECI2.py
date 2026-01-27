# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_HFFE_ECI.py
# SPDX-License-Identifier: MIT
"""
nnUNetTrainer_HFFE_ECI
- Inherit ONLY from nnUNetTrainer (base)
- Build PlainConvUNetWithHFFE_ECI
- HFFE: modifies encoder skips BEFORE decoder
- ECI: injects edge cues INSIDE decoder stages + optional edge supervision

Design goals:
- Minimal coupling to nnUNet version specifics (follow your HFFE/ECI trainer styles)
- Deterministic seed (best-effort) via initialize()
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ------------------------
# Reproducibility helpers (copied from your ECI trainer)
# ------------------------
def _set_global_seed(seed: int,
                     deterministic: bool = True,
                     cudnn_benchmark: bool = False,
                     allow_tf32: bool = False,
                     use_deterministic_algorithms: bool = False) -> None:
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(deterministic)

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


# ------------------------
# Edge GT / loss helpers (copied from your ECI trainer)
# ------------------------
def _pick_target_by_spatial_size(target, hw: Tuple[int, int]) -> torch.Tensor:
    H, W = int(hw[0]), int(hw[1])

    if torch.is_tensor(target):
        t = target
    else:
        assert isinstance(target, (list, tuple))
        for t in target:
            if torch.is_tensor(t) and t.shape[-2:] == (H, W):
                return t
        t = target[0]

    if t.shape[-2:] == (H, W):
        return t

    if t.ndim == 4:  # (B,1,H,W)
        t_ = F.interpolate(t.float(), size=(H, W), mode="nearest")
        return t_.to(dtype=t.dtype)
    if t.ndim == 3:  # (B,H,W)
        t_ = F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
        return t_.to(dtype=t.dtype)
    raise ValueError(f"Unexpected target tensor shape: {tuple(t.shape)}")


def _foreground_mask_from_target(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return (t > 0).unsqueeze(1).float()
    if t.ndim == 4 and t.shape[1] == 1:
        return (t > 0).float()
    if t.ndim == 4 and t.shape[1] > 1:
        return (t[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()
    raise ValueError(f"Unsupported target shape for foreground mask: {tuple(t.shape)}")


def mask_to_edge_band(mask01: torch.Tensor, band_px: int = 1) -> torch.Tensor:
    if band_px <= 0:
        raise ValueError("band_px must be positive.")
    k = 2 * int(band_px) + 1
    pad = int(band_px)

    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=pad)
    edge = (dil - ero).clamp_(0.0, 1.0)
    return edge


def soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * gt).sum(dim=(2, 3)) + eps
    den = (prob + gt).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


class nnUNetTrainer_HFFE_ECI2(nnUNetTrainer):
    # ------------------------
    # Reproducibility (same knobs as your ECI trainer)
    # ------------------------
    seed: int = 2568
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    use_deterministic_algorithms: bool = False

    # ------------------------
    # HFFE hyperparameters (copy your HFFE trainer style)
    # ------------------------
    hffe_enabled = True
    hffe_apply_levels = (0, 1)
    hffe_mode = "concat"  # "replace" | "concat"

    hffe_sam_kernel = 7
    hffe_coord_reduction = 32
    hffe_align_corners = False
    hffe_use_cross_residual = True

    hffe_use_residual_gate = False
    hffe_gate_warmup_epochs = 0
    hffe_gate_ramp_epochs = 10
    hffe_gate_final = 1.0
    hffe_split_fuse_gate = False
    hffe_swm_temperature = 1.0
    hffe_swm_eps = 0.0
    hffe_debug = False

    # ------------------------
    # ECI hyperparameters (copy your ECI trainer style)
    # ------------------------
    # which decoder stages to inject ECI (decoder order: low-res -> high-res), allow negatives
    eci_apply_levels: Optional[Sequence[int]] = (-2, -1)

    # inject scale schedule
    eci_inject_warmup_epochs: int = 0
    eci_inject_max_scale: float = 1.0

    # edge loss schedule
    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 10
    edge_loss_max_weight: float = 0.3
    edge_band_px: int = 1
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0

    # supervise which ECI stages for edge (allow negatives)
    edge_supervision_levels: Optional[Sequence[int]] = (-2, -1)

    def initialize(self):
        _set_global_seed(
            int(self.seed),
            deterministic=bool(self.deterministic),
            cudnn_benchmark=bool(self.cudnn_benchmark),
            allow_tf32=bool(self.allow_tf32),
            use_deterministic_algorithms=bool(self.use_deterministic_algorithms),
        )

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        return super().initialize()

    @classmethod
    def build_network_architecture(cls, *args, **kwargs):
        """
        EXACTLY follow your nnUNetTrainer_HFFE style:
        - supports both training-like and inference-like call signatures
        - resolves pydoc imports for arch_kwargs_requires_import
        - directly instantiates plugin network class to reduce version coupling
        """
        import copy
        import pydoc

        def _resolve_req_imports(arch_kwargs_local, req_import_list):
            arch_kwargs_local = copy.deepcopy(arch_kwargs_local)
            for k in req_import_list:
                if k not in arch_kwargs_local:
                    continue
                v = arch_kwargs_local[k]
                if v is None:
                    continue
                if not isinstance(v, str):
                    continue
                v_str = v.strip()
                if v_str == "":
                    arch_kwargs_local[k] = None
                    continue
                obj = pydoc.locate(v_str)
                if obj is None:
                    raise RuntimeError(f"Failed to import {k}={v_str} via pydoc.locate")
                arch_kwargs_local[k] = obj
            return arch_kwargs_local

        enable_deep_supervision = kwargs.get("enable_deep_supervision", None)

        # Case 1: training-like: (str, dict, list, int, int, bool?)
        if len(args) >= 5 and isinstance(args[0], str):
            arch_kwargs = args[1]
            req_import = args[2]
            num_input_channels = args[3]
            num_segmentation_heads = args[4]

            if enable_deep_supervision is None:
                if len(args) >= 6:
                    enable_deep_supervision = args[5]
                else:
                    enable_deep_supervision = kwargs.get("enable_deep_supervision", False)

            arch_kwargs = _resolve_req_imports(arch_kwargs, req_import)

        # Case 2: inference-like: (dict, list, int, int, enable_deep_supervision=bool)
        elif len(args) >= 4 and isinstance(args[0], dict) and isinstance(args[1], (list, tuple)):
            arch_kwargs = args[0]
            req_import = args[1]
            num_input_channels = args[2]
            num_segmentation_heads = args[3]
            if enable_deep_supervision is None:
                enable_deep_supervision = False
            arch_kwargs = _resolve_req_imports(arch_kwargs, req_import)

        else:
            raise TypeError(
                f"Unexpected build_network_architecture call. args={args}, kwargs={kwargs}. "
                "Expected either (str, dict, list, int, int, bool) or (dict, list, int, int, enable_deep_supervision=bool)."
            )

        # Import your merged network here (late import avoids import-time issues)
        from nnunetv2.network_architecture.plainconvunet_hffe_eci import PlainConvUNetWithHFFE_ECI

        net = PlainConvUNetWithHFFE_ECI(
            input_channels=num_input_channels,
            num_classes=num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs,

            # HFFE knobs
            hffe_enabled=cls.hffe_enabled,
            hffe_apply_levels=cls.hffe_apply_levels,
            hffe_mode=cls.hffe_mode,
            hffe_kwargs=dict(
                sam_kernel=cls.hffe_sam_kernel,
                coord_reduction=cls.hffe_coord_reduction,
                align_corners=cls.hffe_align_corners,
                use_cross_residual=cls.hffe_use_cross_residual,
                use_residual_gate=cls.hffe_use_residual_gate,
                gate_init=0.0 if cls.hffe_use_residual_gate else 1.0,
                split_fuse_gate=cls.hffe_split_fuse_gate,
                debug=cls.hffe_debug,
                swm_temperature=cls.hffe_swm_temperature,
                swm_eps=cls.hffe_swm_eps,
            ),

            # ECI knobs
            eci_apply_levels=cls.eci_apply_levels,
        )

        if cls.hffe_debug and hasattr(net, "set_hffe_debug"):
            net.set_hffe_debug(True)
        if cls.hffe_use_residual_gate and hasattr(net, "set_hffe_gate_scale"):
            net.set_hffe_gate_scale(0.0)

        return net

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # ---- HFFE residual gate schedule (same as your HFFE trainer) ----
        if getattr(self, "network", None) is not None and self.hffe_use_residual_gate:
            e = int(self.current_epoch)
            if e < self.hffe_gate_warmup_epochs:
                s = 0.0
            else:
                if self.hffe_gate_ramp_epochs <= 0:
                    s = float(self.hffe_gate_final)
                else:
                    t = min(1.0, (e - self.hffe_gate_warmup_epochs) / float(self.hffe_gate_ramp_epochs))
                    s = float(self.hffe_gate_final) * t
            if hasattr(self.network, "set_hffe_gate_scale"):
                self.network.set_hffe_gate_scale(s)

        # ---- ECI inject scale schedule (same as your ECI trainer) ----
        if hasattr(self.network, "set_eci_inject_scale"):
            if self.eci_inject_warmup_epochs <= 0:
                s = float(self.eci_inject_max_scale)
            else:
                frac = min(1.0, (self.current_epoch) / float(self.eci_inject_warmup_epochs))
                s = float(self.eci_inject_max_scale) * frac
            self.network.set_eci_inject_scale(s)

    # ------------------------
    # Edge loss (same structure as your ECI trainer, but FIX negative levels)
    # ------------------------
    def _edge_loss_weight(self) -> float:
        if not bool(self.edge_loss_enabled):
            return 0.0
        if self.edge_loss_warmup_epochs <= 0:
            return float(self.edge_loss_max_weight)
        frac = min(1.0, (self.current_epoch + 1) / float(self.edge_loss_warmup_epochs))
        return float(self.edge_loss_max_weight) * frac

    def _compute_edge_loss_from_logits(self, edge_logits: torch.Tensor, target) -> torch.Tensor:
        t = _pick_target_by_spatial_size(target, edge_logits.shape[-2:])
        mask01 = _foreground_mask_from_target(t)
        edge_gt = mask_to_edge_band(mask01, band_px=int(self.edge_band_px))

        bce = F.binary_cross_entropy_with_logits(edge_logits, edge_gt)
        prob = torch.sigmoid(edge_logits)
        dice = soft_dice_loss(prob, edge_gt)

        return float(self.edge_bce_weight) * bce + float(self.edge_dice_weight) * dice

    @staticmethod
    def _normalize_levels(levels: Sequence[int], n: int) -> Sequence[int]:
        out = []
        for i in levels:
            ii = int(i)
            if ii < 0:
                ii = n + ii
            if 0 <= ii < n:
                out.append(ii)
        return out

    def _compute_edge_loss(self, target) -> torch.Tensor:
        if not hasattr(self.network, "get_eci_edge_logits"):
            return torch.zeros((), device=self.device)

        edge_list = self.network.get_eci_edge_logits()
        if not isinstance(edge_list, (list, tuple)) or len(edge_list) == 0:
            return torch.zeros((), device=self.device)

        n = len(edge_list)
        if self.edge_supervision_levels is None:
            levels = [n - 1]
        else:
            levels = list(self._normalize_levels(self.edge_supervision_levels, n))
            if len(levels) == 0:
                levels = [n - 1]

        losses = []
        for i in levels:
            logits = edge_list[i]
            if logits is None:
                continue
            losses.append(self._compute_edge_loss_from_logits(logits, target))
        if len(losses) == 0:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    # ------------------------
    # train_step (same as your ECI trainer; does NOT depend on ECI trainer class)
    # ------------------------
    def train_step(self, batch: dict) -> dict:
        # edge loss disabled -> behave like base trainer
        if not bool(self.edge_loss_enabled):
            if hasattr(self.network, "set_eci_edge_head_enabled"):
                self.network.set_eci_edge_head_enabled(False)
            return super().train_step(batch)

        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        w_edge = float(self._edge_loss_weight())
        if hasattr(self.network, "set_eci_edge_head_enabled"):
            self.network.set_eci_edge_head_enabled(w_edge > 0)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            l_seg = self.loss(output, target)
            l_edge = torch.zeros((), device=self.device)
            if w_edge > 0:
                l_edge = self._compute_edge_loss(target)
            total_loss = l_seg + (w_edge * l_edge)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": float(total_loss.detach().cpu().item()),
            "seg_loss": float(l_seg.detach().cpu().item()),
            "edge_loss": float(l_edge.detach().cpu().item()),
            "w_edge": float(w_edge),
        }
