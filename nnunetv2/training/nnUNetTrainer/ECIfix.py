# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_ECI.py
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

def _extract_label_tensor(target) -> torch.Tensor:
    if torch.is_tensor(target):
        return target
    if isinstance(target, (list, tuple)) and len(target) > 0:
        return target[0]
    raise TypeError(f"Unsupported target type: {type(target)}")

def _pick_target_by_spatial_size(target, hw: Tuple[int, ...]) -> torch.Tensor:
    """
    Select target matching spatial size hw (H, W) or (D, H, W).
    """
    spatial_shape = hw
    
    if torch.is_tensor(target):
        t = target
    else:
        # try to find exact match
        t = target[0]
        for candidate in target:
            if torch.is_tensor(candidate) and candidate.shape[-len(spatial_shape):] == spatial_shape:
                t = candidate
                break

    # check shape match
    if t.shape[-len(spatial_shape):] == spatial_shape:
        return t
        
    # downsample/upsample with nearest
    # Input t is usually (B, 1, spatial...) or (B, spatial...)
    ndim_spatial = len(spatial_shape)
    
    # Pre-process to (B, C, spatial...) for interpolate
    if t.ndim == ndim_spatial + 1: # (B, spatial) -> add channel
        t_in = t.unsqueeze(1).float()
    else: # (B, C, spatial)
        t_in = t.float()

    t_out = F.interpolate(t_in, size=spatial_shape, mode="nearest")
    
    # Post-process back to original dim structure if needed
    if t.ndim == ndim_spatial + 1:
        t_out = t_out.squeeze(1)
        
    return t_out.to(dtype=t.dtype)

def _foreground_mask_from_target(t: torch.Tensor) -> torch.Tensor:
    # returns (B, 1, spatial...)
    if t.ndim == 3: # 2D (B, H, W)
        return (t > 0).unsqueeze(1).float()
    if t.ndim == 4: # 3D (B, D, H, W) or 2D onehot (B, C, H, W)
        # Check if second dim is channel or spatial. nnUNet targets are usually (B, 1, ...) or (B, ...)
        # But safest is checking against typical one-hot behavior. 
        # For simplicity, standard nnUNet target is (B, 1, D, H, W).
        if t.shape[1] == 1:
            return (t > 0).float()
        # Assume one-hot if C > 1
        return (t[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()
    if t.ndim == 5: # 3D (B, 1, D, H, W) typically
        return (t > 0).float()
    raise ValueError(f"Unsupported target shape for foreground mask: {tuple(t.shape)}")

def mask_to_edge_band(mask01: torch.Tensor, band_px: int = 1) -> torch.Tensor:
    """
    Compute morphological edge band. Supports 2D and 3D.
    mask01: (B, 1, H, W) or (B, 1, D, H, W)
    """
    if band_px <= 0:
        raise ValueError("band_px must be positive.")
    k = 2 * int(band_px) + 1
    pad = int(band_px)
    
    dim = mask01.ndim - 2 # exclude B, C
    
    if dim == 2:
        # dilation
        dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)
        # erosion: 1 - dilation(1-mask)
        ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=pad)
    elif dim == 3:
        dil = F.max_pool3d(mask01, kernel_size=k, stride=1, padding=pad)
        ero = 1.0 - F.max_pool3d(1.0 - mask01, kernel_size=k, stride=1, padding=pad)
    else:
        raise ValueError(f"Unsupported spatial dimension: {dim}")

    edge = (dil - ero).clamp_(0.0, 1.0)
    return edge

def soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Sum over spatial dims (2,3) or (2,3,4)
    dims = tuple(range(2, prob.ndim))
    num = 2.0 * (prob * gt).sum(dim=dims) + eps
    den = (prob + gt).sum(dim=dims) + eps
    return 1.0 - (num / den).mean()

class nnUNetTrainer_ECIfix(nnUNetTrainer):
    eci_inject_warmup_epochs: int = 20
    eci_inject_max_scale: float = 1.0

    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 20
    edge_loss_max_weight: float = 0.3
    edge_band_px: int = 1
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0

    edge_supervision_levels: Optional[Sequence[int]] = None

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> nn.Module:
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
        # Align target to edge logits spatial size
        t = _pick_target_by_spatial_size(target, edge_logits.shape[2:])
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
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # Autocast handling for AMP
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            
            # 1. Main Segmentation Loss
            l_seg = self.loss(output, target)
            
            # 2. Auxiliary Edge Loss
            l_edge = torch.zeros((), device=self.device)
            w_edge = self._edge_loss_weight()
            
            if w_edge > 0:
                l_edge = self._compute_edge_loss(target)
                total_loss = l_seg + w_edge * l_edge
            else:
                total_loss = l_seg

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

        return {'loss': l_seg.detach().cpu().numpy(), 'edge_loss': l_edge.detach().cpu().numpy()}