"""
YOLOv8n + Focal Loss 替换 BCE
解决 positive 类别不平衡问题
参考：https://github.com/ultralytics/ultralytics/issues/22167
"""
from typing import Any
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import FocalLoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class FocalDetectionLoss(v8DetectionLoss):
    """用 Focal Loss 替换 BCE 分类损失，alpha 向 positive 倾斜"""
    def __init__(self, model, gamma=1.5, alpha=0.75):
        # alpha=0.75 表示对 positive(少数类) 给予更高权重
        super().__init__(model)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        print(f"[FocalDetectionLoss] gamma={gamma}, alpha={alpha}")

    def __call__(self, preds: Any, batch: dict) -> tuple:
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ── Focal Loss 替换 BCE ──
        loss[1] = self.focal(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # ── Box + DFL 保持不变 ──
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


class FocalTrainer(DetectionTrainer):
    def init_criterion(self):
        return FocalDetectionLoss(self.model, gamma=1.5, alpha=0.75)


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets/brain-tumor/brain-tumor.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="brain_tumor_n_focal",
        augment=False,
        pretrained=True,
        trainer=FocalTrainer,
    )
