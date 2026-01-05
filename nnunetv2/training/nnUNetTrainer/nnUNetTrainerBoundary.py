import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from torch import nn
import torch.nn.functional as F

# ==========================================
# 1. 独立的 Soft Dice Loss (修复版)
# ==========================================
class SafeSoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False):
        super(SafeSoftDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, x, y, loss_mask=None):
        # 1. 如果 y (target) 不是 one-hot，转为 one-hot
        if self.do_bg:
            if y.shape[1] == 1:
                y_onehot = torch.zeros(x.shape, device=x.device)
                y_onehot.scatter_(1, y.long(), 1)
                y = y_onehot
        else:
            if y.shape[1] == 1:
                y_onehot = torch.zeros(x.shape, device=x.device)
                y_onehot.scatter_(1, y.long(), 1)
                y = y_onehot
            
            # 移除背景类 (channel 0)
            x = x[:, 1:]
            y = y[:, 1:]

        # 2. 计算交集和分母
        axes = tuple(range(2, x.ndim))
        if self.batch_dice:
            axes = (0,) + axes # [Batch, Spatial...] -> Sum -> [Channel]

        intersect = (x * y).sum(dim=axes)
        denominator = (x + y).sum(dim=axes)

        # 3. 计算 Dice
        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        
        # 4. 返回 1 - dice
        return 1 - dice.mean()

# ==========================================
# 2. 软边界操作 (Soft Morphology)
# ==========================================
class SoftMorphology(nn.Module):
    def __init__(self, kernel_size=5, dim=2):
        super(SoftMorphology, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x, operation='dilation'):
        if self.dim == 2:
            pool_op = F.max_pool2d
        elif self.dim == 3:
            pool_op = F.max_pool3d
        else:
            raise ValueError("Only 2D and 3D are supported")

        if operation == 'dilation':
            return pool_op(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        elif operation == 'erosion':
            return -pool_op(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        else:
            raise ValueError(f"Unknown operation: {operation}")

# ==========================================
# 3. 边界 Loss (修复版)
# ==========================================
class SoftBoundaryDiceLoss(nn.Module):
    def __init__(self, kernel_size=5, dim=2, smooth=1e-5):
        super(SoftBoundaryDiceLoss, self).__init__()
        self.morph = SoftMorphology(kernel_size, dim)
        self.smooth = smooth

    def forward(self, net_output, target):
        # target 转 one-hot
        with torch.no_grad():
            if target.shape[1] == 1:
                target_onehot = torch.zeros_like(net_output)
                target_onehot.scatter_(1, target.long(), 1)
            else:
                target_onehot = target

        # Softmax 归一化
        pred_softmax = torch.softmax(net_output, dim=1)

        # 提取预测边界
        pred_dilated = self.morph(pred_softmax, 'dilation')
        pred_eroded = self.morph(pred_softmax, 'erosion')
        pred_boundary = pred_dilated - pred_eroded

        # 提取 GT 边界
        gt_dilated = self.morph(target_onehot, 'dilation')
        gt_eroded = self.morph(target_onehot, 'erosion')
        gt_boundary = gt_dilated - gt_eroded

        # 计算边界 Dice
        # axes: (0, 2, 3) -> 在 batch 和空间维度求和，结果是 [Channel]
        axes = tuple(range(2, net_output.ndim))
        axes = (0,) + axes 

        intersect = (pred_boundary * gt_boundary).sum(dim=axes)
        denominator = pred_boundary.sum(dim=axes) + gt_boundary.sum(dim=axes)
        
        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        
        # [修复点]: dice 现在是一维向量 [C]，直接切片 [1:] 即可，不能用 [:, 1:]
        return 1 - dice[1:].mean()

# ==========================================
# 4. 组合 Loss (Dice + CE + Boundary)
# ==========================================
class DC_and_BD_loss(nn.Module):
    def __init__(self, ce_kwargs, bd_kwargs, weight_ce=1, weight_dice=1, weight_bd=1, ignore_label=None):
        super(DC_and_BD_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_bd = weight_bd
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SafeSoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=False)
        self.bd = SoftBoundaryDiceLoss(**bd_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        dc_loss = self.dc(torch.softmax(net_output, dim=1), target) if self.weight_dice != 0 else 0
        
        # 这里的 net_output 传给 bd，bd 内部自己做 softmax
        bd_loss = self.bd(net_output, target) if self.weight_bd != 0 else 0

        total_loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_bd * bd_loss
        return total_loss

# ==========================================
# 5. 自定义 Trainer
# ==========================================
class nnUNetTrainerBoundaryLoss(nnUNetTrainer):
    def _build_loss(self):
        bd_kwargs = {'kernel_size': 5, 'dim': 2}
        
        loss = DC_and_BD_loss(
            {}, 
            bd_kwargs,
            weight_ce=1, 
            weight_dice=1, 
            weight_bd=0.5, 
            ignore_label=self.label_manager.ignore_label
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss