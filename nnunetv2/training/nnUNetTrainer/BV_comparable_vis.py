import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm  # 引入进度条库

# 强制使用非交互式后端，防止 Linux 服务器报错 "Unable to connect to X server"
plt.switch_backend('Agg')

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================
CONFIG = {
    # 路径配置 (已替换为您提供的最新路径)
    "dir_image": "/data0/users/liuxiangdong/data/BV/test/images",
    
    "dir_mask_gt": "/data0/users/liuxiangdong/data/BV/test/masks",
    
    #"dir_mask_gt": "/data0/users/liuxiangdong/data/BV/inferences/baseline8stage",
    "dir_mask_infer": "/data0/users/liuxiangdong/data/BV/inferences/HFFE11",
    "dir_output": "/data0/users/liuxiangdong/data/BV/vis/8stageHFFE11_vs_gt",

    # 颜色配置 (RGB格式)
    "class_colors": {
        1: [255, 0, 0],   # SVC: 红色 (Red)
        2: [0, 255, 0]    # IVC: 绿色 (Green)
    },
    
    # 透明度配置 (Alpha channel)
    "alpha": 0.4,
    
    # 文件扩展名过滤
    "img_ext": ".png"
}

# ==============================================================================
# 文件名映射逻辑
# ==============================================================================
def get_pred_name_rule(filename):
    # 场景 A: 名字完全相同
    return filename
    # 场景 B: 如需修改后缀，在此处调整

# ==============================================================================
# 核心功能模块
# ==============================================================================

def blend_mask(image, mask, colors, alpha):
    """
    将单通道 Mask 以半透明颜色叠加到原图上。
    """
    color_mask = np.zeros_like(image)
    
    for class_id, color in colors.items():
        class_bool_idx = (mask == class_id)
        if np.any(class_bool_idx):
            color_mask[class_bool_idx] = color

    foreground = mask > 0
    output = image.copy()
    
    if np.any(foreground):
        img_fg = image[foreground]
        mask_fg = color_mask[foreground]
        blended_fg = cv2.addWeighted(img_fg, 1 - alpha, mask_fg, alpha, 0)
        output[foreground] = blended_fg
        
    return output

def main():
    # 1. 准备输出目录
    output_dir = Path(CONFIG["dir_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 获取所有原图文件
    image_dir = Path(CONFIG["dir_image"])
    img_files = sorted([f.name for f in image_dir.glob(f"*{CONFIG['img_ext']}")])
    
    if not img_files:
        print(f"错误: 在 {CONFIG['dir_image']} 未找到 {CONFIG['img_ext']} 文件。")
        return

    print(f"准备处理 {len(img_files)} 张图像...")

    # 使用 tqdm 包装循环，显示进度条
    # desc: 进度条左侧描述文字, unit: 单位
    for img_name in tqdm(img_files, desc="Processing", unit="img"):
        
        # 3. 构造完整路径
        path_img = os.path.join(CONFIG["dir_image"], img_name)
        path_gt = os.path.join(CONFIG["dir_mask_gt"], img_name)
        
        pred_name = get_pred_name_rule(img_name)
        path_pred = os.path.join(CONFIG["dir_mask_infer"], pred_name)

        # 4. 检查与读取
        if not os.path.exists(path_img) or not os.path.exists(path_gt):
            # 使用 tqdm.write 可以在不打断进度条的情况下打印信息
            tqdm.write(f"跳过: 文件缺失 - {img_name}")
            continue
            
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        
        if os.path.exists(path_pred):
            pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        else:
            tqdm.write(f"警告: 预测文件不存在 {pred_name}, 使用全黑 Mask 代替。")
            pred = np.zeros_like(gt)

        # 5. 生成叠加
        vis_gt = blend_mask(img, gt, CONFIG["class_colors"], CONFIG["alpha"])
        vis_pred = blend_mask(img, pred, CONFIG["class_colors"], CONFIG["alpha"])

        # 6. 绘图
        # 注意: 创建大量 figure 会消耗内存，这里显式处理
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(vis_gt)
        axes[0].set_title(f"GT: {img_name}", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(vis_pred)
        axes[1].set_title(f"Pred: {pred_name}", fontsize=12)
        axes[1].axis('off')

        plt.suptitle("Red: SVC (Class 1) | Green: IVC (Class 2)", fontsize=14, y=0.95)
        
        # 7. 保存并释放内存
        save_path = output_dir / f"vis_{img_name}"
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig) # 关键：必须关闭，否则内存泄漏

    print("\n所有任务处理完成。")

if __name__ == "__main__":
    main()