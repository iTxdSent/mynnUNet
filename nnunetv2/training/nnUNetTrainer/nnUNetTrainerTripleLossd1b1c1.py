import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ==========================================
# Part 1: 基础工具类 (Soft Morphology & Skeleton)
# ==========================================

class SoftMorphology(nn.Module):
    """
    用于 Boundary Loss 的形态学操作
    """
    def __init__(self, kernel_size=3, dim=2):
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

class SoftSkeletonize(nn.Module):
    """
    用于 clDice 的软骨架化操作 (对应论文 Algorithm 1)
    """
    def __init__(self, num_iter=10):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        # 论文中使用 min_pool (实现为 -max_pool(-x))
        if len(img.shape) == 4: # 2D: (B, C, H, W)
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5: # 3D
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        return img

    def soft_dilate(self, img):
        # 论文中使用 max_pool
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        return img

    def soft_open(self, img):
        # Open = Dilate(Erode(I))
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        # Algorithm 1: Iterative Skeletonization
        skel = F.relu(img - self.soft_open(img))
        for i in range(self.num_iter):
            img = self.soft_erode(img)
            skel = skel + F.relu(img - self.soft_open(img))
        return skel

# ==========================================
# Part 2: 各个 Loss 模块
# ==========================================

class SafeSoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False):
        super(SafeSoftDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, x, y):
        if self.do_bg:
            if y.shape[1] == 1:
                y_onehot = torch.zeros_like(x)
                y_onehot.scatter_(1, y.long(), 1)
                y = y_onehot
        else:
            if y.shape[1] == 1:
                y_onehot = torch.zeros_like(x)
                y_onehot.scatter_(1, y.long(), 1)
                y = y_onehot
            x = x[:, 1:]
            y = y[:, 1:]

        axes = tuple(range(2, x.ndim))
        if self.batch_dice:
            axes = (0,) + axes

        intersect = (x * y).sum(dim=axes)
        denominator = (x + y).sum(dim=axes)
        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()

class SoftBoundaryDiceLoss(nn.Module):
    def __init__(self, kernel_size=3, dim=2, smooth=1e-5):
        super(SoftBoundaryDiceLoss, self).__init__()
        self.morph = SoftMorphology(kernel_size, dim)
        self.smooth = smooth

    def forward(self, net_output, target):
        with torch.no_grad():
            if target.shape[1] == 1:
                target_onehot = torch.zeros_like(net_output)
                target_onehot.scatter_(1, target.long(), 1)
            else:
                target_onehot = target

        pred_softmax = torch.softmax(net_output, dim=1)

        pred_dilated = self.morph(pred_softmax, 'dilation')
        pred_eroded = self.morph(pred_softmax, 'erosion')
        pred_boundary = pred_dilated - pred_eroded

        gt_dilated = self.morph(target_onehot, 'dilation')
        gt_eroded = self.morph(target_onehot, 'erosion')
        gt_boundary = gt_dilated - gt_eroded

        axes = tuple(range(2, net_output.ndim))
        axes = (0,) + axes 

        intersect = (pred_boundary * gt_boundary).sum(dim=axes)
        denominator = pred_boundary.sum(dim=axes) + gt_boundary.sum(dim=axes)
        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        
        # 只取前景类
        return 1 - dice[1:].mean()

class SoftcLDiceLoss(nn.Module):
    def __init__(self, iter_=10, smooth=1.):
        super(SoftcLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeleton = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_true, y_pred):
        # y_true, y_pred 需要是 (B, C, H, W) 且值在 [0,1]
        skel_pred = self.soft_skeleton(y_pred)
        skel_true = self.soft_skeleton(y_true)
        
        # 计算前景类的 Tprec 和 Tsens (跳过背景 channel 0)
        # 保持维度为 [Channel] 以便后续 mean()
        # 注意：这里要在 Spatial + Batch 维度求和，但保留 Channel 维度
        axes = tuple(range(2, y_pred.ndim)) # (2, 3) for 2D
        axes = (0,) + axes # (0, 2, 3)
        
        # Tprec: 预测骨架有多少在 GT Mask 里？
        tprec_num = (skel_pred * y_true).sum(dim=axes) + self.smooth
        tprec_denom = skel_pred.sum(dim=axes) + self.smooth
        tprec = tprec_num / tprec_denom
        
        # Tsens: GT 骨架有多少在 预测 Mask 里？(防断裂核心)
        tsens_num = (skel_true * y_pred).sum(dim=axes) + self.smooth
        tsens_denom = skel_true.sum(dim=axes) + self.smooth
        tsens = tsens_num / tsens_denom
        
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        
        # 1 - clDice, 且只对前景求平均
        return 1.0 - cl_dice[1:].mean()

# ==========================================
# Part 3: 组合 Loss (Triple Loss)
# ==========================================

class Triple_Loss(nn.Module):
    def __init__(self, ce_kwargs, bd_kwargs, cl_kwargs, weight_ce=1, weight_dice=1, weight_bd=1, weight_cl=1, ignore_label=None):
        super(Triple_Loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_bd = weight_bd
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SafeSoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=False)
        self.bd = SoftBoundaryDiceLoss(**bd_kwargs)
        self.cl = SoftcLDiceLoss(**cl_kwargs)

    def forward(self, net_output, target):
        # 1. 准备 Softmax 预测图
        pred_softmax = torch.softmax(net_output, dim=1)
        
        # 2. 准备 One-hot GT
        with torch.no_grad():
            if target.shape[1] == 1:
                target_onehot = torch.zeros_like(net_output)
                target_onehot.scatter_(1, target[:, 0].long().unsqueeze(1), 1)
            else:
                target_onehot = target

        # 3. 计算各项 Loss
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        dc_loss = self.dc(pred_softmax, target) if self.weight_dice != 0 else 0
        bd_loss = self.bd(net_output, target) if self.weight_bd != 0 else 0
        cl_loss = self.cl(target_onehot, pred_softmax) if self.weight_cl != 0 else 0

        total_loss = (self.weight_ce * ce_loss + 
                      self.weight_dice * dc_loss + 
                      self.weight_bd * bd_loss + 
                      self.weight_cl * cl_loss)
        return total_loss

# ==========================================
# Part 4: Trainer
# ==========================================

class nnUNetTrainerTripleLoss(nnUNetTrainer):
    def _build_loss(self):
        # 1. Boundary Loss 参数: kernel=3 (追求直角和细节)
        bd_kwargs = {'kernel_size': 5, 'dim': 2}
        
        # 2. clDice Loss 参数: iter=25 (对应 k 值，因最大半径约 20，设 25 较安全)
        cl_kwargs = {'iter_': 25, 'smooth': 1e-5} 
        
        loss = Triple_Loss(
            {}, # ce_kwargs
            bd_kwargs,
            cl_kwargs,
            weight_ce=1.0, 
            weight_dice=1.0, # 降低普通 Dice 权重
            weight_bd=1.0,   # 强力保形状
            weight_cl=1.0,   # 强力保连接 (防断裂)
            ignore_label=self.label_manager.ignore_label
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss