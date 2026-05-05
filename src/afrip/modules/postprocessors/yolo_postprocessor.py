"""YOLO 风格 objectness-based 后处理组件。"""
from __future__ import annotations

import numpy as np

from afrip.modules.registry import POSTPROCESSORS
from afrip.utils.nms import multiclass_nms


@POSTPROCESSORS.register("YOLOObjectnessPostprocessor")
class YOLOObjectnessPostprocessor:
    """单类 objectness 检测后处理。

    当前模式下类别分支不参与推理，所有候选框统一映射为类别 0。
    """

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
        bboxes: np.ndarray,
        obj_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = np.where(obj_scores >= self.conf_thresh)[0]
        bboxes = bboxes[keep]
        scores = obj_scores[keep]
        labels = np.zeros(len(scores), dtype=np.int32)

        if len(bboxes) == 0:
            return bboxes, scores, labels

        finite = np.isfinite(bboxes).all(axis=1)
        proper = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        mask = finite & proper
        bboxes = bboxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(bboxes) == 0:
            return bboxes, scores, labels

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
        return bboxes, scores, labels