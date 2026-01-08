"""
nnUNetV2 Trainer: 2D-only MCI (Seg + Edge + per-class CenterLine heatmaps)

Requested improvements implemented:
- Loss weights: lambda_edge=0.3, lambda_cl=0.1 (can be overridden via configuration_manager)
- Warmup: first W epochs set aux losses to 0, then linearly ramp to target over R epochs
- Edge GT: per-class boundary, fixed narrow band (1–2 px), preserves touching-class boundaries
- Edge loss: BCEWithLogits with dynamic pos_weight (per-batch) + soft Dice (binary)
- CL GT: per-class heatmap proxy; supervised only on top K highest-resolution outputs (default K=2)
- Deep supervision target alignment: if target is list, use target[i]; else NN downsample per output
- 2D only (Conv2d)
- Robust autocast (uses torch.autocast) and safe AllGatherGrad when dist not initialized

Place this file at:
  nnunetv2/training/nnUNetTrainer/mciTrainer.py
and run:
  nnUNetv2_train <DATASET_ID> 2d <FOLD> -tr nnUNetTrainerMCI
"""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, autocast
import torch.distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.helpers import dummy_context

# Import your mounted MCI network wrapper (adjust path if you placed it elsewhere)
from nnunetv2.nets.mci_network import MCIUNetWrapper2D


Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Patch nnUNet Dice AllGatherGrad to be safe when torch.distributed is not initialized.
# Some nnUNet forks call AllGatherGrad even in single-process mode when batch_dice=True.
# -----------------------------------------------------------------------------
class _SafeAllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            gathered = [torch.zeros_like(tensor) for _ in range(world)]
            dist.all_gather(gathered, tensor)
            ctx.world_size = world
            ctx.rank = dist.get_rank()
            return torch.stack(gathered, 0)
        ctx.world_size = 1
        ctx.rank = 0
        return tensor.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
            return grad_output.squeeze(0)
        g = grad_output.contiguous()
        dist.all_reduce(g)
        return g[ctx.rank]

try:
    import nnunetv2.training.loss.dice as _nnunet_dice_mod
    _nnunet_dice_mod.AllGatherGrad = _SafeAllGatherGrad
except Exception:
    pass


# -----------------------------------------------------------------------------
# Small morphology helpers (2D)
# -----------------------------------------------------------------------------
def _dilate2d(x: Tensor, k: int = 3) -> Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _erode2d(x: Tensor, k: int = 3) -> Tensor:
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def edge_from_multiclass(seg: Tensor, num_classes: int, *, band_px: int = 1) -> Tensor:
    """
    seg: (B,1,H,W) int labels
    Returns: (B,1,H,W) binary edge map.

    Key property: preserves boundaries between adjacent foreground classes
    by computing edges per class and combining with max.

    band_px=1: inner boundary ring (mask - erode(mask))
    band_px=2: thicker ring (dilate(mask) - erode(mask))
    """
    seg = seg.long()
    B, _, H, W = seg.shape
    edges: List[Tensor] = []
    k = 3  # fixed to produce ~1px neighborhood; thickness controlled by formula

    for c in range(1, num_classes):
        m = (seg == c).float()  # (B,1,H,W)
        if band_px <= 1:
            e = (m - _erode2d(m, k=k)).clamp(0.0, 1.0)
        else:
            e = (_dilate2d(m, k=k) - _erode2d(m, k=k)).clamp(0.0, 1.0)
        edges.append(e)

    if len(edges) == 0:
        return torch.zeros((B, 1, H, W), device=seg.device, dtype=torch.float32)
    return torch.stack(edges, dim=0).max(dim=0).values


def cl_heatmap_per_class(seg: Tensor, num_classes: int, *, iters: int, k: int = 3) -> Tensor:
    """
    Per-class "soft centerline" proxy via iterative erosion accumulation.

    seg: (B,1,H,W) int labels
    returns: (B, C-1, H, W) float in [0,1]
    """
    seg = seg.long()
    B, _, H, W = seg.shape
    C = max(1, num_classes - 1)
    out = torch.zeros((B, C, H, W), device=seg.device, dtype=torch.float32)

    iters = int(max(1, iters))
    # cap by feature size to avoid vanishing
    iters = min(iters, max(1, min(H, W) // 8))

    for c in range(1, num_classes):
        m = (seg == c).float()
        if torch.count_nonzero(m) == 0:
            continue
        dist_map = torch.zeros_like(m)
        cur = m.clone()
        for _ in range(iters):
            dist_map = dist_map + cur
            cur = _erode2d(cur, k=k)
            cur = (cur > 0.5).float()
            if torch.count_nonzero(cur) == 0:
                break
        max_val = dist_map.flatten(1).amax(dim=1).view(B, 1, 1, 1)
        dist_map = dist_map / (max_val + 1e-6)
        out[:, c - 1:c] = dist_map
    return out


def soft_dice_loss_binary_from_logits(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    logits: (B,1,H,W), target: (B,1,H,W) float {0,1}
    """
    p = torch.sigmoid(logits)
    p = p.contiguous().view(p.shape[0], -1)
    t = target.contiguous().view(target.shape[0], -1)

    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class nnUNetTrainerMCI2(nnUNetTrainer):
    # ---------------------------
    # Build & mount network
    # ---------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        base_net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        # guard: 2D only
        if hasattr(base_net, "encoder") and hasattr(base_net.encoder, "conv_op"):
            if base_net.encoder.conv_op is torch.nn.Conv3d:
                raise RuntimeError("nnUNetTrainerMCI is 2D-only, but base network uses Conv3d.")
        return MCIUNetWrapper2D(base_net, num_classes=int(num_output_channels), deep_supervision=bool(enable_deep_supervision))

    # ---------------------------
    # Init
    # ---------------------------
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device)

        # Main seg loss (nnUNet style)
        self.loss_seg = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {},
            weight_ce=1.0,
            weight_dice=1.0
        )

        # Edge loss parts
        self.loss_edge_dice = soft_dice_loss_binary_from_logits
        self.edge_band_px = int(getattr(self.configuration_manager, "edge_band_px", 1))  # 1 or 2 recommended

        # CL loss
        self.loss_cl = nn.MSELoss()

        # Target weights (what you will set to 0.3 / 0.1)
        self.lambda_edge_target = float(getattr(self.configuration_manager, "lambda_edge", 0.3))
        self.lambda_cl_target = float(getattr(self.configuration_manager, "lambda_cl", 0.2))

        # Warmup & ramp
        self.aux_warmup_epochs = int(getattr(self.configuration_manager, "aux_warmup_epochs", 20))
        self.aux_ramp_epochs = int(getattr(self.configuration_manager, "aux_ramp_epochs", 10))  # 0 = step to target

        # CL settings
        self.cl_max_iters = int(getattr(self.configuration_manager, "cl_max_iters", 20))
        self.cl_supervise_levels = int(getattr(self.configuration_manager, "cl_supervise_levels", 1))  # supervise top 1-2
        self.cl_kernel = int(getattr(self.configuration_manager, "cl_kernel", 3))

    # ---------------------------
    # Disable torch.compile (stability)
    # ---------------------------
    def _do_i_compile(self):
        return False

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_ds_weights(self, n_outputs: int) -> Tensor:
        w = getattr(self, "ds_loss_weights", None)
        if w is None:
            return torch.ones(n_outputs, device=self.device, dtype=torch.float32) / float(n_outputs)
        w = torch.tensor(w, device=self.device, dtype=torch.float32)
        if w.numel() != n_outputs:
            return torch.ones(n_outputs, device=self.device, dtype=torch.float32) / float(n_outputs)
        return w

    def _ensure_target_list(self, target: Union[Tensor, List[Tensor]], outputs: List[Tensor]) -> List[Tensor]:
        if isinstance(target, list):
            return target
        tgt_list: List[Tensor] = []
        for out in outputs:
            _, _, H, W = out.shape
            t = F.interpolate(target.float(), size=(H, W), mode='nearest').long()
            tgt_list.append(t)
        return tgt_list

    def _aux_weights_for_epoch(self) -> Tuple[float, float]:
        """
        Warmup + ramp schedule for aux losses, based on self.current_epoch (nnUNet uses 0-indexed epochs).
        """
        e = int(getattr(self, "current_epoch", 0))
        if e < self.aux_warmup_epochs:
            return 0.0, 0.0
        if self.aux_ramp_epochs <= 0:
            return self.lambda_edge_target, self.lambda_cl_target
        t = min(1.0, float(e - self.aux_warmup_epochs + 1) / float(self.aux_ramp_epochs))
        return self.lambda_edge_target * t, self.lambda_cl_target * t

    def _scaled_cl_iters(self, ds_level: int, hw: Tuple[int, int]) -> int:
        """
        Scale CL erosion iterations for deep supervision outputs.
        - Start from cl_max_iters at highest resolution (ds_level=0)
        - Reduce by 2^ds_level
        - Cap by feature map size to avoid vanishing on small maps
        """
        h, w = int(hw[0]), int(hw[1])
        base = max(1, int(self.cl_max_iters) // (2 ** int(ds_level)))
        size_cap = max(1, min(h, w) // 8)
        return int(min(base, size_cap))

    def _edge_bce_with_posweight(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCEWithLogits with per-batch pos_weight to counter extreme imbalance.
        """
        # target is float {0,1}
        pos = target.sum()
        neg = target.numel() - pos
        # pos_weight = neg/pos; clamp to avoid exploding gradients early
        pw = (neg / (pos + 1.0)).clamp(1.0, 50.0).to(dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

    # ---------------------------
    # Train/Val
    # ---------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # move to device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        lam_edge, lam_cl = self._aux_weights_for_epoch()

        self.optimizer.zero_grad(set_to_none=True)

        with (autocast(device_type='cuda', enabled=True) if self.device.type == 'cuda' else dummy_context()):
            seg_outs, edge_outs, cl_outs = self.network(data, return_aux=True)

            tgt_list = self._ensure_target_list(target, seg_outs)
            weights = self._get_ds_weights(len(seg_outs))

            total_seg = torch.zeros((), device=self.device, dtype=torch.float32)
            total_edge = torch.zeros((), device=self.device, dtype=torch.float32)
            total_cl = torch.zeros((), device=self.device, dtype=torch.float32)

            n_classes = int(self.label_manager.num_segmentation_heads)

            for i, (seg_logits, edge_logits, cl_logits) in enumerate(zip(seg_outs, edge_outs, cl_outs)):
                seg_gt = tgt_list[i].long()

                # seg loss
                total_seg = total_seg + weights[i] * self.loss_seg(seg_logits, seg_gt)

                # edge loss (always computed, but multiplied by lam_edge outside)
                edge_gt = edge_from_multiclass(seg_gt, n_classes, band_px=self.edge_band_px)
                edge_bce = self._edge_bce_with_posweight(edge_logits, edge_gt)
                edge_dice = self.loss_edge_dice(edge_logits, edge_gt)
                total_edge = total_edge + weights[i] * (edge_bce + edge_dice)

                # CL loss: only supervise top K resolution outputs
                if i < self.cl_supervise_levels:
                    iters_i = self._scaled_cl_iters(i, seg_logits.shape[-2:])
                    cl_gt = cl_heatmap_per_class(
                        seg_gt, n_classes,
                        iters=iters_i,
                        k=self.cl_kernel
                    )
                    total_cl = total_cl + weights[i] * self.loss_cl(cl_logits, cl_gt)

            loss = total_seg + lam_edge * total_edge + lam_cl * total_cl

        # backward (match nnUNet: GradScaler on cuda)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': float(loss.detach().cpu()),
            'loss_seg': float(total_seg.detach().cpu()),
            'loss_edge': float(total_edge.detach().cpu()),
            'loss_cl': float(total_cl.detach().cpu()),
            'lam_edge': float(lam_edge),
            'lam_cl': float(lam_cl),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        lam_edge, lam_cl = self._aux_weights_for_epoch()

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        with torch.no_grad():
            with (autocast(device_type='cuda', enabled=True) if self.device.type == 'cuda' else dummy_context()):
                seg_outs, edge_outs, cl_outs = self.network(data, return_aux=True)

                tgt_list = self._ensure_target_list(target, seg_outs)
                weights = self._get_ds_weights(len(seg_outs))

                total_seg = torch.zeros((), device=self.device, dtype=torch.float32)
                total_edge = torch.zeros((), device=self.device, dtype=torch.float32)
                total_cl = torch.zeros((), device=self.device, dtype=torch.float32)

                n_classes = int(self.label_manager.num_segmentation_heads)

                for i, (seg_logits, edge_logits, cl_logits) in enumerate(zip(seg_outs, edge_outs, cl_outs)):
                    seg_gt = tgt_list[i].long()
                    total_seg = total_seg + weights[i] * self.loss_seg(seg_logits, seg_gt)

                    edge_gt = edge_from_multiclass(seg_gt, n_classes, band_px=self.edge_band_px)
                    edge_bce = self._edge_bce_with_posweight(edge_logits, edge_gt)
                    edge_dice = self.loss_edge_dice(edge_logits, edge_gt)
                    total_edge = total_edge + weights[i] * (edge_bce + edge_dice)

                    if i < self.cl_supervise_levels:
                        iters_i = self._scaled_cl_iters(i, seg_logits.shape[-2:])
                        cl_gt = cl_heatmap_per_class(seg_gt, n_classes, iters=iters_i, k=self.cl_kernel)
                        total_cl = total_cl + weights[i] * self.loss_cl(cl_logits, cl_gt)

                loss = total_seg + lam_edge * total_edge + lam_cl * total_cl

            # --- Online evaluation (match nnUNet expectations) ---
            output = seg_outs[0] if self.enable_deep_supervision else seg_outs
            gt_for_eval = tgt_list[0] if self.enable_deep_supervision else tgt_list

            axes = [0] + list(range(2, output.ndim))
            if self.label_manager.has_regions:
                predicted_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if self.label_manager.has_regions:
                    mask = ~gt_for_eval[:, -1:]
                else:
                    mask = 1 - gt_for_eval[:, -1:]
                    gt_for_eval = gt_for_eval[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_onehot, gt_for_eval, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        return {
            'loss': float(loss.detach().cpu()),
            'loss_seg': float(total_seg.detach().cpu()),
            'loss_edge': float(total_edge.detach().cpu()),
            'loss_cl': float(total_cl.detach().cpu()),
            'lam_edge': float(lam_edge),
            'lam_cl': float(lam_cl),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }
