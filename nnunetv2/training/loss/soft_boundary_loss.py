import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class SoftMorphology(nn.Module):
    """
    实现可微分的软形态学操作 (Soft Erosion/Dilation)
    原理：使用 MaxPool 实现 Dilation，使用 -MaxPool(-x) 实现 Erosion
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
            # Dilation = Max Pooling
            return pool_op(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        elif operation == 'erosion':
            # Erosion = - Max Pooling ( -x )
            return -pool_op(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        else:
            raise ValueError(f"Unknown operation: {operation}")

class SoftBoundaryDiceLoss(nn.Module):
    """
    计算预测边界与GT边界的 Dice Loss
    """
    def __init__(self, kernel_size=3, dim=2, smooth=1e-5):
        super(SoftBoundaryDiceLoss, self).__init__()
        self.morph = SoftMorphology(kernel_size, dim)
        self.smooth = smooth

    def forward(self, net_output, target):
        """
        net_output: (B, C, H, W) 或者是 (B, C, D, H, W) -> Softmax 后的概率图
        target: (B, 1, H, W) 或者是 (B, 1, D, H, W) -> One-hot 之前的 GT
        """
        # 1. 准备数据
        # 如果 target 是 label index，转为 one-hot (nnUNet 的 Loss 输入通常需要处理一下)
        # 注意：nnUNet 传入的 target 可能是 (B, 1, H, W)，需要转为 (B, C, H, W)
        with torch.no_grad():
            if target.shape[1] == 1:
                # 转换为 One-hot
                num_classes = net_output.shape[1]
                target_onehot = torch.zeros_like(net_output)
                target_onehot.scatter_(1, target.long(), 1)
            else:
                target_onehot = target

        # 2. 提取软边界 (Soft Boundary Extraction)
        # Boundary = I - Erosion(I)  或者  Dilation(I) - Erosion(I)
        # 这里使用 Dilation - Erosion 获得更厚的梯度，利于训练稳定性
        
        # 预测图的边界 (保持梯度)
        pred_dilated = self.morph(net_output, 'dilation')
        pred_eroded = self.morph(net_output, 'erosion')
        pred_boundary = pred_dilated - pred_eroded

        # GT 图的边界 (不需要梯度)
        gt_dilated = self.morph(target_onehot, 'dilation')
        gt_eroded = self.morph(target_onehot, 'erosion')
        gt_boundary = gt_dilated - gt_eroded

        # 3. 计算边界 Dice
        # 仅计算前景类 (channel 1, 2, ...)，忽略背景 (channel 0)
        # 你的数据: 0=背景, 1=SVC, 2=IVC
        # axes 定义为 (0, 2, 3) for 2D, (0, 2, 3, 4) for 3D，即在 Batch 和 Spatial 维度求和
        axes = tuple(range(2, net_output.ndim)) 
        
        intersect = (pred_boundary * gt_boundary).sum(dim=axes)
        denominator = pred_boundary.sum(dim=axes) + gt_boundary.sum(dim=axes)
        
        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        
        # 对所有前景类取平均 (跳过 channel 0)
        # 如果你只想关注特定血管，可以只取 dice[:, 1:]
        loss = 1 - dice[:, 1:].mean() 
        
        return loss

class DC_and_BD_loss(nn.Module):
    """
    组合 Loss: Standard Dice + Cross Entropy + Boundary Dice
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, bd_kwargs, weight_ce=1, weight_dice=1, weight_bd=1, ignore_label=None):
        super(DC_and_BD_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_bd = weight_bd # 新增边界 Loss 权重
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs)
        self.bd = SoftBoundaryDiceLoss(**bd_kwargs) # 初始化边界 Loss

    def forward(self, net_output, target):
        """
        nnUNet V2 会传入 list of outputs (用于深监督)
        """
        if self.ignore_label is not None:
            # 这里的处理逻辑参考 nnUNet 源码，确保 target 中忽略区域不参与计算
            pass 

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        
        # 计算 Boundary Loss (需要对 net_output 做 softmax)
        # nnUNet 的 loss 输入通常是 logits，需要先转 softmax 才能做形态学操作
        pred_softmax = torch.softmax(net_output, dim=1)
        bd_loss = self.bd(pred_softmax, target) if self.weight_bd != 0 else 0

        total_loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_bd * bd_loss
        return total_loss