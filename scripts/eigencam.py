import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── 加载模型 ──────────────────────────────────────────
model = YOLO("runs/brain_tumor_n_noaug/weights/best.pt")
torch_model = model.model
torch_model.eval()

# ── 选取目标层（YOLOv8 backbone 末端）────────────────
target_layers = [torch_model.model[9]]

# ── 随机选 4 张验证集图片 ─────────────────────────────
val_dir = Path("datasets/brain-tumor/images/val")
all_images = list(val_dir.glob("*.jpg"))
random.seed(42)
sample_paths = random.sample(all_images, 4)

# ── reshape_transform（适配 YOLOv8 特征图格式）────────
def yolov8_reshape_transform(x):
    # YOLOv8 的特征图是 list，取最后一个
    if isinstance(x, (list, tuple)):
        x = x[-1]
    return x

# ── 逐张生成热力图 ─────────────────────────────────────
Path("outputs/eigencam").mkdir(parents=True, exist_ok=True)
results = []

for img_path in sample_paths:
    # 读图并归一化
    img_bgr = cv2.imread(str(img_path))
    img_bgr = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    # 转 tensor
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)

    # EigenCAM
    cam = EigenCAM(
        model=torch_model,
        target_layers=target_layers,
        reshape_transform=yolov8_reshape_transform
    )
    grayscale_cam = cam(input_tensor=tensor)[0]

    # 叠加热力图
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    results.append((img_path.name, img_rgb, cam_image))
    print(f"✅ 完成: {img_path.name}")

# ── 拼成 2x4 对比图（原图 | 热力图）──────────────────
fig, axes = plt.subplots(4, 2, figsize=(10, 20))
fig.suptitle("EigenCAM: Brain Tumor Detection - Model Attention Visualization",
             fontsize=13, fontweight='bold', y=0.98)

for i, (name, orig, cam_img) in enumerate(results):
    axes[i, 0].imshow(orig)
    axes[i, 0].set_title(f"Original: {name}", fontsize=8)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(cam_img)
    axes[i, 1].set_title("EigenCAM Heatmap", fontsize=8)
    axes[i, 1].axis('off')

plt.tight_layout()
output_path = "outputs/eigencam/eigencam_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 热力图保存在: {output_path}")
