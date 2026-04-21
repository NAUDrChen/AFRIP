from __future__ import annotations

import numpy as np


def nms(bboxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> list[int]:
    """Pure Python NMS，返回保留框的索引列表。"""
    valid_mask = np.isfinite(bboxes).all(axis=1)
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    if bboxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(1e-10, xx2 - xx1) * np.maximum(1e-10, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(
    scores: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    nms_thresh: float,
    num_classes: int,
    class_agnostic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多类 NMS，返回 (scores, labels, bboxes) 过滤后的结果。"""
    if class_agnostic:
        keep = nms(bboxes, scores, nms_thresh)
        if len(keep) == 0:
            return (np.zeros((0,), dtype=scores.dtype),
                    np.zeros((0,), dtype=labels.dtype),
                    np.zeros((0, 4), dtype=bboxes.dtype))
        return scores[keep], labels[keep], bboxes[keep]

    out_scores, out_labels, out_bboxes = [], [], []
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_keep = nms(bboxes[inds], scores[inds], nms_thresh)
        if len(c_keep) == 0:
            continue
        out_scores.append(scores[inds][c_keep])
        out_labels.append(np.full(len(c_keep), i, dtype=labels.dtype))
        out_bboxes.append(bboxes[inds][c_keep])

    if not out_scores:
        return (np.zeros((0,), dtype=scores.dtype),
                np.zeros((0,), dtype=labels.dtype),
                np.zeros((0, 4), dtype=bboxes.dtype))

    scores = np.concatenate(out_scores)
    labels = np.concatenate(out_labels)
    bboxes = np.concatenate(out_bboxes)
    order = scores.argsort()[::-1]
    return scores[order], labels[order], bboxes[order]
