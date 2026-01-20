# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_HFFE_ECI_UGHI_Pure.py
from __future__ import annotations

import copy
import pydoc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.network_architecture.plainconvunet_hffe_eci_ughi_pure import PlainConvUNetWithHFFE_ECI_PureUGHILite
from nnunetv2.custom_modules.eci_lite import ECILiteConfig
from nnunetv2.custom_modules.ughi_pure import PureUGHILiteConfig


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

    if t.ndim == 4:
        return F.interpolate(t.float(), size=(H, W), mode="nearest").to(dtype=t.dtype)
    if t.ndim == 3:
        return F.interpolate(t.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).to(dtype=t.dtype)
    raise ValueError(f"Unexpected target shape: {tuple(t.shape)}")


def _foreground_mask_from_target(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return (t > 0).unsqueeze(1).float()
    if t.ndim == 4 and t.shape[1] == 1:
        return (t > 0).float()
    if t.ndim == 4 and t.shape[1] > 1:
        return (t[:, 1:, ...].sum(dim=1, keepdim=True) > 0).float()
    raise ValueError(f"Unsupported target shape for foreground mask: {tuple(t.shape)}")


def mask_to_edge_band(mask01: torch.Tensor, band_px: int = 1) -> torch.Tensor:
    k = 2 * int(band_px) + 1
    pad = int(band_px)
    dil = F.max_pool2d(mask01, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask01, kernel_size=k, stride=1, padding=pad)
    return (dil - ero).clamp_(0.0, 1.0)


def soft_dice_loss(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * gt).sum(dim=(2, 3)) + eps
    den = (prob + gt).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()

class _SegPlusEdgeLoss(nn.Module):
    """
    Wrap the original nnUNet loss so base trainer can keep producing tp_hard/fp_hard/fn_hard,
    while total loss = seg_loss + w_edge * edge_loss.
    """
    def __init__(self, seg_loss: nn.Module, trainer_ref):
        super().__init__()
        self.seg_loss = seg_loss
        self.trainer_ref = trainer_ref

    def forward(self, net_output, target):
        loss_seg = self.seg_loss(net_output, target)

        w = float(self.trainer_ref._edge_loss_weight())
        if w <= 0:
            return loss_seg

        loss_edge = self.trainer_ref._compute_edge_loss(target)
        return loss_seg + (w * loss_edge)


class nnUNetTrainer_HFFE_ECI_UGHI_Pure(nnUNetTrainer):
    # -----------------------------
    # HFFE：只用于生成 H_k 给 UGHI（不改写 skip）
    # -----------------------------
    hffe_enabled: bool = True
    hffe_apply_levels: Sequence[int] = (0, 1)  # encoder levels
    hffe_kwargs = dict(
        sam_kernel=7,
        coord_reduction=32,
        align_corners=False,
        use_cross_residual=True,
        use_residual_gate=False,
        gate_init=1.0,
        split_fuse_gate=False,
        debug=False,
        swm_temperature=1.0,
        swm_eps=0.0,
    )
    

    # -----------------------------
    # ECI：插在 decoder stage 后
    # -----------------------------
    eci_apply_levels: Sequence[int] = (1,2,3)   # None -> all decoder stages
    eci_cfg = ECILiteConfig()

    # ECI 注入强度 warmup（可选）
    eci_inject_warmup_epochs: int = 20
    eci_inject_max_scale: float = 1.0

    # -----------------------------
    # UGHI：作用在 decoder stage 输入的竖向 up 部分
    # -----------------------------
    ughi_enabled: bool = True
    ughi_apply_levels: Sequence[int] = (0,1)  # None -> all decoder stages
    ughi_cfg = PureUGHILiteConfig(
    use_edge_gate=True,
    detach_prev_edge_gate=True,   # 关键：避免跨stage反传导致图膨胀/显存爆
    inject_scale_init=0.10,
    align_corners=False,
    )

    
    # -----------------------------
    # Edge Loss（真正加入总 loss）
    # -----------------------------
    edge_loss_enabled: bool = True
    edge_loss_warmup_epochs: int = 20
    edge_loss_max_weight: float = 0.3
    edge_band_px: int = 1
    edge_bce_weight: float = 1.0
    edge_dice_weight: float = 1.0
    edge_supervision_levels: Optional[Sequence[int]] = None  # None -> highest-res stage only

    def initialize(self):
        super().initialize()

        # keep original seg loss
        self._seg_loss = self.loss

        n_dec = len(self.network.decoder.stages)


        # replace loss with wrapped loss (seg + edge)
        self.loss = _SegPlusEdgeLoss(self._seg_loss, self)


    @classmethod
    def build_network_architecture(cls, *args, **kwargs):
        """
        兼容 nnUNet v2 训练/推理两种 build_network_architecture 调用风格（与你 nnUNetTrainer_HFFE.py 一致）
        """
        enable_deep_supervision = kwargs.get("enable_deep_supervision", None)

        # Case 1: training style: (str, dict, list, int, int, bool?)
        if len(args) >= 5 and isinstance(args[0], str):
            arch_kwargs = _resolve_req_imports(args[1], args[2])
            num_input_channels = args[3]
            num_segmentation_heads = args[4]
            if enable_deep_supervision is None:
                enable_deep_supervision = args[5] if len(args) >= 6 else kwargs.get("enable_deep_supervision", False)

        # Case 2: inference style: (dict, list, int, int, enable_deep_supervision=bool)
        elif len(args) >= 4 and isinstance(args[0], dict):
            arch_kwargs = _resolve_req_imports(args[0], args[1])
            num_input_channels = args[2]
            num_segmentation_heads = args[3]
            if enable_deep_supervision is None:
                enable_deep_supervision = False
        else:
            raise TypeError(f"Unexpected build_network_architecture call. args={args}, kwargs={kwargs}")

        net = PlainConvUNetWithHFFE_ECI_PureUGHILite(
            input_channels=num_input_channels,
            num_classes=num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs,
            # HFFE
            hffe_enabled=cls.hffe_enabled,
            hffe_apply_levels=cls.hffe_apply_levels,
            hffe_kwargs=cls.hffe_kwargs,
            # ECI
            eci_cfg=cls.eci_cfg,
            eci_apply_levels=cls.eci_apply_levels,
            # UGHI
            ughi_enabled=cls.ughi_enabled,
            ughi_apply_levels=cls.ughi_apply_levels,
            ughi_cfg=cls.ughi_cfg,
        )
        return net

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # ECI inject scale schedule
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

        # 默认只监督最高分辨率（最后一个 stage）
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

    # -----------------------------
    # 关键：把 edge loss 加入总 loss
    # -----------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        # target 可能是 list（deep supervision），里面 tensor 也需要在 device
        if torch.is_tensor(target):
            target = target.to(self.device, non_blocking=True)
        elif isinstance(target, (list, tuple)):
            target = [t.to(self.device, non_blocking=True) for t in target]

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = bool(getattr(self, "enable_amp", False)) and (self.device.type == "cuda")
        scaler = getattr(self, "grad_scaler", None)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = self.network(data)
            loss_seg = self.loss(out, target)

            w_edge = self._edge_loss_weight()
            loss_edge = self._compute_edge_loss(target) if w_edge > 0 else torch.zeros_like(loss_seg)

            loss = loss_seg + float(w_edge) * loss_edge

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_seg": loss_seg.detach().cpu().numpy(),
            "loss_edge": loss_edge.detach().cpu().numpy(),
            "w_edge": float(w_edge),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if torch.is_tensor(target):
            target = target.to(self.device, non_blocking=True)
        elif isinstance(target, (list, tuple)):
            target = [t.to(self.device, non_blocking=True) for t in target]

        use_amp = bool(getattr(self, "enable_amp", False)) and (self.device.type == "cuda")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            out = self.network(data)
            loss_seg = self.loss(out, target)

            w_edge = self._edge_loss_weight()
            loss_edge = self._compute_edge_loss(target) if w_edge > 0 else torch.zeros_like(loss_seg)
            loss = loss_seg + float(w_edge) * loss_edge

        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_seg": loss_seg.detach().cpu().numpy(),
            "loss_edge": loss_edge.detach().cpu().numpy(),
            "w_edge": float(w_edge),
        }
