"""
YOLOv8n + Dropout2d 在分类头最后一个无BN的Conv2d之前插入
观察：cv3每个scale的最后一层是裸Conv2d(无BN无激活)，无任何正则化
原理：Dropout2d随机关掉整个特征通道，强迫分类头学习更鲁棒特征
     在无BN层之前插入，规避BN+Dropout方差偏移问题
参考：Li et al., 2019 "Understanding the Disharmony between Dropout and BN"
"""
import torch.nn as nn
from ultralytics import YOLO


def insert_dropout_to_detect_head(model, p=0.1):
    """在 Detect head cv3 每个 scale 的最后 Conv2d 前插入 Dropout2d"""
    detect = model.model.model[-1]
    for i, seq in enumerate(detect.cv3):
        # seq[-1] 是裸 Conv2d（无BN），在它前面插入 Dropout2d
        new_seq = nn.Sequential(
            seq[0],                    # Conv+BN+SiLU
            seq[1],                    # Conv+BN+SiLU
            nn.Dropout2d(p=p),         # ← 新增：随机关掉整个通道
            seq[2],                    # 裸 Conv2d（无BN）
        )
        detect.cv3[i] = new_seq
    print(f"✅ Dropout2d(p={p}) 已插入 cv3 x3 scales")
    return model


if __name__ == "__main__":
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DropoutTrainer(DetectionTrainer):
        def build_model(self, cfg=None, weights=None, verbose=True):
            model = super().build_model(cfg=cfg, weights=weights, verbose=verbose)
            insert_dropout_to_detect_head(model, p=0.1)
            return model

    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets/brain-tumor/brain-tumor.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="brain_tumor_n_dropout",
        augment=False,
        pretrained=True,
        trainer=DropoutTrainer,
    )
