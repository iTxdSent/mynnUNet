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

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

# =================================================================================================
# [FIX] Deterministic Loss Implementation
# 解决 nnUNet 默认 2D CrossEntropy 在 CUDA 上无法确定性复现的问题
# =================================================================================================
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
# 👇 [新增] 必须引入这个 Dice Loss 类
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss 

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    替换标准 nn.CrossEntropyLoss。
    通过将 (B, C, H, W) 展平为 (B*H*W, C) 再计算，避开 nll_loss2d 的非确定性实现。
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 如果 target 是 (B, 1, H, W) 形式，先压缩为 (B, H, W)
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
            
        # 展平操作：将空间维度全部展平到 Batch 维度
        if input.ndim > 2:
            # input: (B, C, H, W) -> (B, H, W, C) -> (N, C)
            input = input.permute(0, *range(2, input.ndim), 1).flatten(0, -2)
            # target: (B, H, W) -> (N)
            target = target.flatten()
            
        return super().forward(input, target)

class DeterministicDCandCELoss(DC_and_CE_loss):
    # 👇 [修改] dice_class 默认值不能是 None，必须设为 MemoryEfficientSoftDiceLoss
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, 
                         ignore_label, dice_class)
        
        # 用 RobustCrossEntropyLoss 覆盖默认的 CE
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
# =================================================================================================

def _set_global_seed(seed: int,
                     deterministic: bool = True,
                     cudnn_benchmark: bool = False,
                     allow_tf32: bool = False,
                     use_deterministic_algorithms: bool = False) -> None:
    """Best-effort reproducibility setup.

    Notes:
      - Full bitwise determinism is not guaranteed if torch.compile is enabled, certain CUDA kernels are used,
        or data augmentation runs in multi-process mode with independent RNG streams.
      - This function fixes the common sources of randomness (python/numpy/torch) and configures CUDA/CuDNN knobs.
    """
    seed = int(seed)

    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN / TF32
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

    # cuBLAS reproducibility (must be set before the first CUDA matmul in many cases; still helpful here)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Hard deterministic algorithms (may raise if an op has no deterministic implementation)
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)



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


class nnUNetTrainer_ECI_0122_35(nnUNetTrainer):
    # ------------------------
    # Reproducibility
    # ------------------------
    seed: int = 2568
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    use_deterministic_algorithms: bool = True  # set True only if you can tolerate potential runtime errors

    # ------------------------
    # Hyperparameters (adjust at class level or in __init__)
    # ------------------------
    # ECI feature injection schedule
    eci_inject_warmup_epochs: int = 0
    eci_inject_max_scale: float = 1.0

    # Edge supervision schedule
    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 10
    edge_loss_max_weight: float = 0.2
    edge_band_px: int = 1  # morphological edge band width
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0

    # Which decoder stages to supervise for edge:
    # By default supervise only the highest-resolution stage to keep it lightweight.
    edge_supervision_levels: Optional[Sequence[int]] = [-2,-1]  # -1 -> [last_stage]

    # ------------------------
    # Reproducibility controls
    # ------------------------
    seed: int = 2568
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    use_deterministic_algorithms: bool = True

    def initialize(self):
        """
        Set global RNG seeds and determinism switches before nnUNet builds dataloaders/augmenters.
        Do NOT reseed every epoch; we seed once so RNG progression remains reproducible.
        """
         # 1. 强制设置确定性标志
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 2. 开启 PyTorch 严格确定性模式
        # 注意：如果你的网络包含不支持确定性的算子（如部分 Upsample 模式），这里会报错
        # 如果报错，请将 upsample mode 改为 'nearest' 或 'bilinear' (align_corners=True/False 需测试)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass # 极老版本没有这个函数


        _set_global_seed(
            int(self.seed),
            deterministic=bool(self.deterministic),
            cudnn_benchmark=bool(self.cudnn_benchmark),
            allow_tf32=bool(self.allow_tf32),
            use_deterministic_algorithms=bool(self.use_deterministic_algorithms),
            #use_deterministic_algorithms=False,
        )

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        return super().initialize()

    def _build_loss(self):
        """
        重写：使用确定性的 DeterministicDCandCELoss 替换默认 Loss
        """
        # 1. 设置 Dice 和 CE 参数
        loss_kwargs = {}
        if self.configuration_manager.batch_dice:
            loss_kwargs['batch_dice'] = self.configuration_manager.batch_dice
            
        # 2. 实例化确定性 Loss (这是替换的关键)
        loss = DeterministicDCandCELoss(
            loss_kwargs, 
            {}, 
            weight_ce=1, 
            weight_dice=1, 
            ignore_label=self.label_manager.ignore_label
        )
        
        # 3. 如果启用了深监督，必须用 Wrapper 包裹它
        # 这样 train_step 里传进来的 list[Tensor] 才能被正确处理
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # 计算深监督权重 (从 nnUNet 源码复制的标准逻辑)
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            weights = [float(i) for i in weights]
            
            loss = DeepSupervisionWrapper(loss, weights)
            
        self.loss = loss
        

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
        *args, **kwargs
    ) -> nn.Module:
        # 强制使用你的自定义网络
        architecture_class_name = "nnunetv2.network_architecture.plainconvunet_eci.PlainConvUNetWithECI"

        # 这里 *args/**kwargs 是为了兼容 nnUNet v2 不同版本基类传入的额外参数（第7个参数等）
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
            levels = []
            for i in self.edge_supervision_levels:
                ii = int(i)
                if ii < 0:
                    ii = len(edge_list) + ii
                if 0 <= ii < len(edge_list):
                    levels.append(ii)
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