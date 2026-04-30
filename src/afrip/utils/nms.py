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


def soft_nms(
    bboxes: np.ndarray,
    scores: np.ndarray,
    nms_thresh: float,
    method: str = "linear",
    sigma: float = 0.5,
    score_thresh: float = 1e-3,
    topk: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS.

    Returns:
        (keep_indices, updated_scores)
    """
    if bboxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), scores

    boxes = bboxes.astype(np.float32, copy=True)
    scores = scores.astype(np.float32, copy=True)
    indices = np.arange(boxes.shape[0], dtype=np.int64)
    keep: list[int] = []

    while scores.size > 0:
        max_idx = int(np.argmax(scores))
        best_box = boxes[max_idx]
        best_score = float(scores[max_idx])
        best_orig_idx = int(indices[max_idx])

        if best_score < score_thresh:
            break

        keep.append(best_orig_idx)
        if topk is not None and len(keep) >= topk:
            break

        boxes = np.delete(boxes, max_idx, axis=0)
        scores = np.delete(scores, max_idx, axis=0)
        indices = np.delete(indices, max_idx, axis=0)
        if scores.size == 0:
            break

        xx1 = np.maximum(best_box[0], boxes[:, 0])
        yy1 = np.maximum(best_box[1], boxes[:, 1])
        xx2 = np.minimum(best_box[2], boxes[:, 2])
        yy2 = np.minimum(best_box[3], boxes[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_best = np.maximum(0.0, best_box[2] - best_box[0]) * np.maximum(0.0, best_box[3] - best_box[1])
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        iou = inter / (area_best + areas - inter + 1e-14)

        if method == "linear":
            decay = np.ones_like(iou)
            high = iou > nms_thresh
            decay[high] -= iou[high]
        elif method == "gaussian":
            decay = np.exp(-(iou * iou) / max(sigma, 1e-6))
        else:
            decay = np.ones_like(iou)
            decay[iou > nms_thresh] = 0.0

        scores *= decay
        valid = scores >= score_thresh
        boxes = boxes[valid]
        scores = scores[valid]
        indices = indices[valid]

    return np.asarray(keep, dtype=np.int64), scores


def multiclass_nms(
    scores: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    nms_thresh: float,
    num_classes: int,
    class_agnostic: bool = False,
    nms_type: str = "hard",
    soft_nms_method: str = "linear",
    soft_nms_sigma: float = 0.5,
    soft_nms_score_thresh: float = 1e-3,
    topk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多类 NMS，返回 (scores, labels, bboxes) 过滤后的结果。"""
    use_soft_nms = nms_type.lower() == "soft"

    if class_agnostic:
        if use_soft_nms:
            keep, _ = soft_nms(
                bboxes,
                scores,
                nms_thresh,
                method=soft_nms_method,
                sigma=soft_nms_sigma,
                score_thresh=soft_nms_score_thresh,
                topk=topk,
            )
        else:
            keep = np.asarray(nms(bboxes, scores, nms_thresh), dtype=np.int64)
            if topk is not None:
                keep = keep[:topk]
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
        if use_soft_nms:
            c_keep, _ = soft_nms(
                bboxes[inds],
                scores[inds],
                nms_thresh,
                method=soft_nms_method,
                sigma=soft_nms_sigma,
                score_thresh=soft_nms_score_thresh,
                topk=topk,
            )
        else:
            c_keep = np.asarray(nms(bboxes[inds], scores[inds], nms_thresh), dtype=np.int64)
            if topk is not None:
                c_keep = c_keep[:topk]
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
    if topk is not None:
        order = order[:topk]
    return scores[order], labels[order], bboxes[order]
