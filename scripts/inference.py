from ultralytics import YOLO
from pathlib import Path
import random

model = YOLO("runs/brain_tumor_n_noaug/weights/best.pt")

val_images = list(Path("datasets/brain-tumor/images/val").glob("*.jpg"))
print(f"验证集共 {len(val_images)} 张图片")

random.seed(42)
sample_images = random.sample(val_images, min(20, len(val_images)))

results = model.predict(
    source=sample_images,
    save=True,
    save_txt=True,
    conf=0.25,
    project="outputs",
    name="inference_results"
)

print(f"\n✅ 推理完成！")
print(f"结果保存在: outputs/inference_results/")

total_detections = 0
for r in results:
    total_detections += len(r.boxes)
print(f"共检测到 {total_detections} 个目标")
