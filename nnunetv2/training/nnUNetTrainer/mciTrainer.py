"""
nnUNetV2 Trainer with 2D-only MCI (Seg + Edge + per-class CenterLine heatmaps).

Implements the user-requested fixes:
1) Network is actually used: build_network_architecture wraps base network and replaces decoder.
2) Edge supervision: 1-channel logits + BCEWithLogits + soft Dice (binary).
3) Deep supervision target alignment: if target is list, use target[i] directly (no interpolate).
4) CL cue: per-class (foreground classes only, channels = C-1).
5) 2D-only operators (Conv2d / pooling2d).
"""

from __future__ import annotations

from typing import List, Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

from nnunetv2.nets.mci_network import MCIUNetWrapper2D


Tensor = torch.Tensor


def soft_dice_loss_binary_from_logits(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    logits: (B,1,H,W)
    target: (B,1,H,W) float in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _dilate2d(x: Tensor, k: int = 3) -> Tensor:
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)


def _erode2d(x: Tensor, k: int = 3) -> Tensor:
    # erosion via -maxpool(-x)
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def edge_from_multiclass(seg: Tensor, num_classes: int, *, k: int = 3) -> Tensor:
    """
    seg: (B,1,H,W) integer labels
    returns: (B,1,H,W) binary edge map.
    Strategy: compute per-class morphological gradient and combine by max.
    This preserves boundaries between adjacent classes (unlike union foreground).
    """
    seg = seg.long()
    B = seg.shape[0]
    edges = []
    for c in range(1, num_classes):  # foreground classes
        m = (seg == c).float()
        dil = _dilate2d(m, k=k)
        ero = _erode2d(m, k=k)
        e = (dil - ero).clamp(0.0, 1.0)
        edges.append(e)
    if len(edges) == 0:
        return torch.zeros((B, 1, seg.shape[-2], seg.shape[-1]), device=seg.device, dtype=torch.float32)
    edge = torch.stack(edges, dim=0).max(dim=0).values
    return edge


def cl_heatmap_per_class(seg: Tensor, num_classes: int, *, iters: int, k: int = 3) -> Tensor:
    """
    Per-class "soft centerline" proxy based on iterative erosion accumulation.

    seg: (B,1,H,W) integer labels
    returns: (B, C-1, H, W) float in [0,1]
    """
    seg = seg.long()
    B, _, H, W = seg.shape
    out = torch.zeros((B, max(1, num_classes - 1), H, W), device=seg.device, dtype=torch.float32)

    # safety: never erode more than a small fraction of image size
    iters = int(max(1, iters))
    iters = min(iters, max(1, min(H, W) // 6))

    for c in range(1, num_classes):
        m = (seg == c).float()  # (B,1,H,W)
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

        # normalize per-sample per-class to [0,1]
        max_val = dist_map.flatten(1).amax(dim=1).view(B, 1, 1, 1)  # (B,1,1,1)
        dist_map = dist_map / (max_val + 1e-6)
        out[:, c - 1:c, :, :] = dist_map

    return out


class nnUNetTrainerMCI(nnUNetTrainer):
    # ---------------------------
    # (1) Build & mount network
    # ---------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """
        Wrap the base nnUNet network and replace its decoder with MCI decoder.
        This ensures optimizer includes MCI parameters (constructed before optimizer setup).
        """
        base_net = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        num_classes = int(num_output_channels)
        deep_supervision = bool(enable_deep_supervision)
        # 2D-only guard: if encoder conv op is Conv3d, raise early.
        if hasattr(base_net, "encoder") and hasattr(base_net.encoder, "conv_op"):
            if base_net.encoder.conv_op is torch.nn.Conv3d:
                raise RuntimeError("This MCI trainer/network is 2D-only, but base encoder is Conv3d.")

        return MCIUNetWrapper2D(base_net, num_classes=num_classes, deep_supervision=deep_supervision)

    # ---------------------------
    # Losses & hyperparameters
    # ---------------------------
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device=torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json,
                         device=device)

        # Main segmentation loss: keep nnUNet default style
        self.loss_seg = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {},
            weight_ce=1.0,
            weight_dice=1.0
        )

        # Edge: binary BCE + Dice from logits
        self.edge_bce = nn.BCEWithLogitsLoss()

        # CL: regression heatmap (MSE)
        self.loss_cl = nn.MSELoss()

        # task weights
        self.lambda_edge = float(getattr(self.configuration_manager, "lambda_edge", 0.5))
        self.lambda_cl = float(getattr(self.configuration_manager, "lambda_cl", 0.25))

        # CL erosion iterations at highest resolution; DS levels scale down
        self.cl_max_iters = int(getattr(self.configuration_manager, "cl_max_iters", 20))

    # ---------------------------
    # Helpers
    # ---------------------------
    def _do_i_compile(self):
        """Disable torch.compile for this custom trainer.

        nnUNetTrainer may call torch.compile on the network and Dice loss. In some environments
        this can fail with FakeTensor device propagation errors (CPU vs CUDA) when using custom
        wrappers/architectures. Disabling compilation keeps behavior correct and stable."""
        return False

    def _get_ds_weights(self, n_outputs: int) -> Tensor:
        # nnUNetTrainer typically sets self.ds_loss_weights when deep supervision is enabled
        w = getattr(self, "ds_loss_weights", None)
        if w is None:
            return torch.ones(n_outputs, device=self.device) / float(n_outputs)
        w = torch.tensor(w, device=self.device, dtype=torch.float32)
        if w.numel() != n_outputs:
            # fall back to equal if mismatch (robustness)
            return torch.ones(n_outputs, device=self.device) / float(n_outputs)
        return w

    def _ensure_target_list(self, target: Union[Tensor, List[Tensor]], outputs: List[Tensor]) -> List[Tensor]:
        """
        If target is already a list (deep supervision GT), return as-is.
        Otherwise, generate DS targets by nearest-neighbor downsampling to each output resolution.
        """
        if isinstance(target, list):
            return target

        # target is (B,1,H,W) labels
        tgt_list = []
        for out in outputs:
            _, _, H, W = out.shape
            t = F.interpolate(target.float(), size=(H, W), mode='nearest').long()
            tgt_list.append(t)
        return tgt_list

    # ---------------------------
    # Train/Val steps
    # ---------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']  # can be list or tensor

        # Move to device (nnUNet default train_step does this; we must replicate it)
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Autocast only on CUDA (match nnUNetTrainer behavior)
        from torch.cuda.amp import autocast
        from nnunetv2.utilities.helpers import dummy_context
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_outs, edge_outs, cl_outs = self.network(data, return_aux=True)

            # targets aligned high->low (list order must match seg_outs)
            tgt_list = self._ensure_target_list(target, seg_outs)
            weights = self._get_ds_weights(len(seg_outs))

            total_seg = 0.0
            total_edge = 0.0
            total_cl = 0.0

            for i, (seg_logits, edge_logits, cl_logits) in enumerate(zip(seg_outs, edge_outs, cl_outs)):
                seg_gt = tgt_list[i].long()

                # Seg loss: use nnUNet's configured loss function at each scale
                total_seg = total_seg + weights[i] * self.loss_seg(seg_logits, seg_gt)

                # Edge GT: per-class boundary bands merged with max (preserves touching-class boundaries)
                edge_gt = edge_from_multiclass(seg_gt, self.label_manager.num_segmentation_heads, k=self.edge_k)
                total_edge = total_edge + weights[i] * (
                    self.edge_bce(edge_logits, edge_gt) + soft_dice_loss_binary_from_logits(edge_logits, edge_gt)
                )

                # CL GT: per-class soft centerline proxy (iters scaled with DS level and feature map size)
                iters_i = self._scaled_cl_iters(i, seg_logits.shape[-2:])
                cl_gt = cl_heatmap_per_class(seg_gt, num_classes=self.label_manager.num_segmentation_heads, iters=iters_i)
                total_cl = total_cl + weights[i] * self.loss_cl(cl_logits, cl_gt)

            loss = total_seg + self.lambda_edge * total_edge + self.lambda_cl * total_cl

        # Backprop: follow nnUNetTrainer (GradScaler if available, gradient clipping)
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
            'loss': loss.detach().cpu().numpy(),
            'loss_seg': float(total_seg.detach().cpu().numpy()),
            'loss_edge': float(total_edge.detach().cpu().numpy()),
            'loss_cl': float(total_cl.detach().cpu().numpy()),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        from torch.cuda.amp import autocast
        from nnunetv2.utilities.helpers import dummy_context
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                seg_outs, edge_outs, cl_outs = self.network(data, return_aux=True)

                tgt_list = self._ensure_target_list(target, seg_outs)
                weights = self._get_ds_weights(len(seg_outs))

                total_seg = 0.0
                total_edge = 0.0
                total_cl = 0.0

                for i, (seg_logits, edge_logits, cl_logits) in enumerate(zip(seg_outs, edge_outs, cl_outs)):
                    seg_gt = tgt_list[i].long()

                    total_seg = total_seg + weights[i] * self.loss_seg(seg_logits, seg_gt)
                    edge_gt = edge_from_multiclass(seg_gt, self.label_manager.num_segmentation_heads, k=self.edge_k)
                    total_edge = total_edge + weights[i] * (
                        self.edge_bce(edge_logits, edge_gt) + soft_dice_loss_binary_from_logits(edge_logits, edge_gt)
                    )

                    iters_i = self._scaled_cl_iters(i, seg_logits.shape[-2:])
                    cl_gt = cl_heatmap_per_class(seg_gt, self.label_manager.num_segmentation_heads, iters=iters_i)
                    total_cl = total_cl + weights[i] * self.loss_cl(cl_logits, cl_gt)

                loss = total_seg + self.lambda_edge * total_edge + self.lambda_cl * total_cl

            # --- Online evaluation (match nnUNetTrainer expectations) ---
            output = seg_outs[0] if self.enable_deep_supervision else seg_outs
            gt_for_eval = tgt_list[0] if self.enable_deep_supervision else tgt_list

            axes = [0] + list(range(2, output.ndim))
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if self.label_manager.has_regions:
                    mask = ~gt_for_eval[:, -1:]
                else:
                    # target can be long or bool
                    if gt_for_eval.dtype == torch.bool:
                        mask = ~gt_for_eval[:, -1:]
                    else:
                        mask = 1 - gt_for_eval[:, -1:]
                    gt_for_eval = gt_for_eval[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, gt_for_eval, axes=axes, mask=mask)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if not self.label_manager.has_regions:
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'loss_seg': float(total_seg.detach().cpu().numpy()),
            'loss_edge': float(total_edge.detach().cpu().numpy()),
            'loss_cl': float(total_cl.detach().cpu().numpy()),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }
