# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_HFFE_ECI.py
# SPDX-License-Identifier: MIT
"""
nnUNetTrainer_HFFE_ECI
- Uses PlainConvUNetWithHFFEAndECI
- Adds optional auxiliary edge loss (same as nnUNetTrainer_ECI)

This trainer is intended for:
- HFFE + ECI joint deployment experiments
- Ablations by toggling edge loss / HFFE enable flags via class attributes
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# Reuse helpers from nnUNetTrainer_ECI
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_ECI import (
    _pick_target_by_spatial_size,
    _foreground_mask_from_target,
    mask_to_edge_band,
    soft_dice_loss,
)


class nnUNetTrainer_HFFE_ECI(nnUNetTrainer):
    # ------------------------
    # Hyperparameters
    # ------------------------
    # HFFE schedule (only effective if your HFFE exposes set_gate_scale)
    hffe_enabled: bool = True
    hffe_gate_warmup_epochs: int = 0
    hffe_gate_max_scale: float = 1.0

    # ECI feature injection schedule
    eci_inject_warmup_epochs: int = 20
    eci_inject_max_scale: float = 1.0

    # Edge supervision schedule
    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 20
    edge_loss_max_weight: float = 0.1
    edge_band_px: int = 1
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0
    edge_supervision_levels: Optional[Sequence[int]] = None  # None -> [last_stage]

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        architecture_class_name = "nnunetv2.network_architecture.plainconvunet_hffe_eci.PlainConvUNetWithHFFEAndECI"
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # HFFE enable toggle
        if hasattr(self.network, "hffe_enabled"):
            self.network.hffe_enabled = bool(self.hffe_enabled)

        # HFFE gate schedule
        if hasattr(self.network, "set_hffe_gate_scale"):
            if self.hffe_gate_warmup_epochs <= 0:
                s = float(self.hffe_gate_max_scale)
            else:
                frac = min(1.0, (self.current_epoch + 1) / float(self.hffe_gate_warmup_epochs))
                s = float(self.hffe_gate_max_scale) * frac
            self.network.set_hffe_gate_scale(s)

        # ECI inject schedule
        if hasattr(self.network, "set_eci_inject_scale"):
            if self.eci_inject_warmup_epochs <= 0:
                s = float(self.eci_inject_max_scale)
            else:
                frac = min(1.0, (self.current_epoch) / float(self.eci_inject_warmup_epochs))
                s = float(self.eci_inject_max_scale) * frac
            self.network.set_eci_inject_scale(s)

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

    def _compute_edge_loss(self, target) -> torch.Tensor:
        if not hasattr(self.network, "get_eci_edge_logits"):
            return torch.zeros((), device=self.device)

        edge_list = self.network.get_eci_edge_logits()
        if not isinstance(edge_list, (list, tuple)) or len(edge_list) == 0:
            return torch.zeros((), device=self.device)

        if self.edge_supervision_levels is None:
            levels = [len(edge_list) - 1]
        else:
            levels = [int(i) for i in self.edge_supervision_levels if 0 <= int(i) < len(edge_list)]
            if len(levels) == 0:
                levels = [len(edge_list) - 1]

        losses = []
        for i in levels:
            logits = edge_list[i]
            if logits is None:
                continue
            losses.append(self._compute_edge_loss_from_logits(logits, target))
        if len(losses) == 0:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def train_step(self, batch: dict) -> dict:
        """Use nnUNet base implementation (keeps AMP/compile compatibility)."""
        return super().train_step(batch)


