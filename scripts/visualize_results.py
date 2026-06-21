from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 结果图片路径
result_dir = Path("runs/detect/outputs/inference_results")
images = list(result_dir.glob("*.jpg"))[:9]  # 取前9张

print(f"找到 {len(images)} 张结果图")

# 拼成 3x3 九宫格
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("YOLOv8 Brain Tumor Detection Results", 
             fontsize=20, fontweight='bold', y=0.98)

for i, ax in enumerate(axes.flatten()):
    if i < len(images):
        img = cv2.imread(str(images[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(images[i].name, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("outputs/detection_grid.png", 
            dpi=150, bbox_inches='tight')
print("✅ 九宫格展示图已保存: outputs/detection_grid.png")
