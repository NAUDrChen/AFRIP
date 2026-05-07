"""YOLO 风格 objectness-based 后处理组件。"""
from __future__ import annotations

import torch

from afrip.models.registry import POSTPROCESSORS
from afrip.utils.nms import multiclass_nms


@POSTPROCESSORS.register("YOLOObjectnessPostprocessor")
class YOLOObjectnessPostprocessor:
    """单类 objectness 检测后处理。"""

    def __init__(
        self,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        nms_type: str = "hard",
        soft_nms_method: str = "linear",
        soft_nms_sigma: float = 0.5,
        soft_nms_score_thresh: float = 1e-3,
        topk: int | None = None,
        class_agnostic: bool = True,
        num_classes: int = 1,
    ):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nms_type = nms_type
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_score_thresh = soft_nms_score_thresh
        self.topk = topk
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes

    def __call__(
        self,
        bboxes: torch.Tensor,
        obj_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        keep = torch.where(obj_scores >= self.conf_thresh)[0]
        bboxes = bboxes[keep]
        scores = obj_scores[keep]
        labels = torch.zeros((scores.shape[0],), dtype=torch.long, device=scores.device)

        if bboxes.numel() == 0:
            return {
                "boxes": bboxes.reshape(0, 4),
                "scores": scores.reshape(0),
                "labels": labels.reshape(0),
            }

        finite = torch.isfinite(bboxes).all(dim=1)
        proper = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        mask = finite & proper
        bboxes = bboxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if bboxes.numel() == 0:
            return {
                "boxes": bboxes.reshape(0, 4),
                "scores": scores.reshape(0),
                "labels": labels.reshape(0),
            }

        scores, labels, bboxes = multiclass_nms(
            scores,
            labels,
            bboxes,
            self.nms_thresh,
            self.num_classes,
            class_agnostic=self.class_agnostic,
            nms_type=self.nms_type,
            soft_nms_method=self.soft_nms_method,
            soft_nms_sigma=self.soft_nms_sigma,
            soft_nms_score_thresh=self.soft_nms_score_thresh,
            topk=self.topk,
        )
        return {
            "boxes": bboxes,
            "scores": scores,
            "labels": labels,
        }