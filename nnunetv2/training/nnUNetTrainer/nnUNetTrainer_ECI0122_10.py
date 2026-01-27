# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ECI.py
# SPDX-License-Identifier: MIT
"""
nnUNetTrainer_ECI
- Uses PlainConvUNetWithECI (ECI-Lite inside decoder stages)
- Adds an auxiliary edge loss (optional but enabled by default here)

Assumptions:
- Based on nnUNet v2 trainer API:
    - train_step expects batch['data'], batch['target'].
    - self.loss computes the main segmentation loss (deep supervision aware).
    - self.network is a torch.nn.Module.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def _extract_label_tensor(target) -> torch.Tensor:
    """
    target may be:
      - torch.Tensor (B,1,H,W) or (B,H,W) with class indices
      - list/tuple of tensors for deep supervision
    We return a tensor chosen later by shape match; here we just validate types.
    """
    if torch.is_tensor(target):
        return target
    if isinstance(target, (list, tuple)) and len(target) > 0:
        # return first for now; selection by shape is done elsewhere
        return target[0]
    raise TypeError(f"Unsupported target type: {type(target)}")


def _pick_target_by_spatial_size(target, hw: Tuple[int, int]) -> torch.Tensor:
    """
    Select the target tensor whose (H,W) matches hw. If none matches, downsample
    the highest-res target to hw with nearest neighbor.
    """
    H, W = int(hw[0]), int(hw[1])

    if torch.is_tensor(target):
        t = target
    else:
        assert isinstance(target, (list, tuple))
        # try to find exact match
        for t in target:
            if torch.is_tensor(t) and t.shape[-2:] == (H, W):
                return t
        t = target[0]

    if t.shape[-2:] == (H, W):
        return t
    # downsample/upsample with nearest so labels remain integers
    if t.ndim == 4:  # (B,1,H,W)
        t_ = F.interpolate(t.float(), size=(H, W), mode="nearest")
        return t_.to(dtype=t.dtype)
    if t.ndim == 3:  # (B,H,W)
        t_ = F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1)
        return t_.to(dtype=t.dtype)
    raise ValueError(f"Unexpected target tensor shape: {tuple(t.shape)}")


def _foreground_mask_from_target(t: torch.Tensor) -> torch.Tensor:
    """
    Convert target labels to a binary foreground mask in {0,1} with shape (B,1,H,W).
    - If t is (B,H,W): treat >0 as foreground.
    - If t is (B,1,H,W): treat >0 as foreground.
    """
    if t.ndim == 3:
        return (t > 0).unsqueeze(1).float()
    if t.ndim == 4 and t.shape[1] == 1:
        return (t > 0).float()
    # If one-hot (B,C,H,W), use foreground union across C>0
    if t.ndim == 4 and t.shape[1] > 1:
        return (t[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()
    raise ValueError(f"Unsupported target shape for foreground mask: {tuple(t.shape)}")


def mask_to_edge_band(mask01: torch.Tensor, band_px: int = 1) -> torch.Tensor:
    """
    Compute a morphological edge band from a binary mask (B,1,H,W) in {0,1}.
    Edge band = dilation(mask) - erosion(mask). This is robust and reproducible.
    """
    if band_px <= 0:
        raise ValueError("band_px must be positive.")
    k = 2 * int(band_px) + 1
    pad = int(band_px)

    # dilation: maxpool(mask)
    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)
    # erosion: 1 - dilation(1-mask)
    ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=pad)
    edge = (dil - ero).clamp_(0.0, 1.0)
    return edge


def soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    prob, gt: (B,1,H,W) floats in [0,1]
    """
    num = 2.0 * (prob * gt).sum(dim=(2, 3)) + eps
    den = (prob + gt).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


class nnUNetTrainer_ECI_0122_10(nnUNetTrainer):
    # ------------------------
    # Hyperparameters (adjust at class level or in __init__)
    # ------------------------
    # ECI feature injection schedule
    eci_inject_warmup_epochs: int = 0
    eci_inject_max_scale: float = 1.0

    # Edge supervision schedule
    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 20
    edge_loss_max_weight: float = 0.3
    edge_band_px: int = 1  # morphological edge band width
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0

    # Which decoder stages to supervise for edge:
    # By default supervise only the highest-resolution stage to keep it lightweight.
    edge_supervision_levels: Optional[Sequence[int]] = [-1]  # None -> [last_stage]

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
        # Force using our custom architecture, independent of what plans say.
        architecture_class_name = "nnunetv2.network_architecture.plainconvunet_eci.PlainConvUNetWithECI"
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

        # 1) ECI inject scale schedule
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
        # align target to edge logits size
        t = _pick_target_by_spatial_size(target, edge_logits.shape[-2:])
        mask01 = _foreground_mask_from_target(t)
        edge_gt = mask_to_edge_band(mask01, band_px=int(self.edge_band_px))

        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(edge_logits, edge_gt)

        # Dice on probabilities
        prob = torch.sigmoid(edge_logits)
        dice = soft_dice_loss(prob, edge_gt)

        return float(self.edge_bce_weight) * bce + float(self.edge_dice_weight) * dice

    def _compute_edge_loss(self, target) -> torch.Tensor:
        if not hasattr(self.network, "get_eci_edge_logits"):
            return torch.zeros((), device=self.device)

        edge_list = self.network.get_eci_edge_logits()
        if not isinstance(edge_list, (list, tuple)) or len(edge_list) == 0:
            return torch.zeros((), device=self.device)

        # Select levels
        if self.edge_supervision_levels is None:
            levels = [len(edge_list) - 1]  # highest resolution stage
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
        # ====== 模式A：完全对齐旧版本实际训练行为（旧版就是 super().train_step）======
        if not bool(self.edge_loss_enabled):
            # 可选但建议：关掉 edge head，避免 decoder wrapper 请求 edge logits
            if hasattr(self.network, "set_eci_edge_head_enabled"):
                self.network.set_eci_edge_head_enabled(False)
            return super().train_step(batch)

        # ====== 模式B：启用 edge loss（保留你现在的新逻辑）======
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        w_edge = float(self._edge_loss_weight())
        if hasattr(self.network, 'set_eci_edge_head_enabled'):
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
            'loss': float(total_loss.detach().cpu().item()),
            'seg_loss': float(l_seg.detach().cpu().item()),
            'edge_loss': float(l_edge.detach().cpu().item()),
            'w_edge': float(w_edge),
        }

