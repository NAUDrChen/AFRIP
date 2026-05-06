"""Typed contracts between dataset, model, matcher, loss and evaluator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class DetectionTarget:
    """Per-image target contract.

    boxes: normalized cxcywh tensor shaped [N, 4]
    labels: class index tensor shaped [N]
    """

    boxes: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DetectionTarget":
        return cls(
            boxes=data["boxes"],
            labels=data["labels"],
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "boxes": self.boxes,
            "labels": self.labels,
        }


@dataclass
class DetectionBatch:
    """Batch contract returned by dataset collate_fn.

    images: [B, C, H, W]
    targets: list of per-image DetectionTarget
    raw_boxes: [M, 7], columns=(batch_idx, class_id, obj_id, x1, y1, x2, y2)
    batch_meta: length-B metadata records
    """

    images: torch.Tensor
    targets: list[DetectionTarget]
    raw_boxes: torch.Tensor
    batch_meta: list[dict[str, Any]]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DetectionBatch":
        return cls(
            images=data["images"],
            targets=normalize_detection_targets(data["targets"]),
            raw_boxes=data["raw_boxes"],
            batch_meta=list(data.get("batch_meta", [])),
        )


@dataclass
class DetectionModelOutput:
    """Dense detection training output contract.

    pred_obj: [B, M, 1] objectness logits
    pred_box: [B, M, 4] decoded xyxy boxes in pixels
    strides: per-level strides, length=L
    feature_shapes: per-level feature map shapes [(H, W), ...], length=L
    extras: optional auxiliary tensors keyed by string
    """

    pred_obj: torch.Tensor
    pred_box: torch.Tensor
    strides: list[int]
    feature_shapes: list[tuple[int, int]]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def stride(self) -> int:
        return int(self.strides[-1])

    @property
    def fmp_size(self) -> tuple[int, int]:
        return tuple(self.feature_shapes[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "pred_obj": self.pred_obj,
            "pred_box": self.pred_box,
            "stride": self.stride,
            "fmp_size": self.fmp_size,
            "strides_all": list(self.strides),
            "fmp_sizes_all": list(self.feature_shapes),
            **self.extras,
        }


@dataclass
class DetectionDetections:
    """Postprocessed inference detections.

    boxes: [N, 4] xyxy in pixels
    scores: [N]
    labels: [N]
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device | None = None) -> "DetectionDetections":
        tensor_device = device if device is not None else torch.device("cpu")
        return cls(
            boxes=torch.zeros((0, 4), dtype=torch.float32, device=tensor_device),
            scores=torch.zeros((0,), dtype=torch.float32, device=tensor_device),
            labels=torch.zeros((0,), dtype=torch.long, device=tensor_device),
        )

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.boxes.detach().cpu().numpy(),
            self.scores.detach().cpu().numpy(),
            self.labels.detach().cpu().numpy(),
        )


def normalize_detection_targets(
    targets: list[DetectionTarget | dict[str, Any]],
) -> list[DetectionTarget]:
    normalized: list[DetectionTarget] = []
    for target in targets:
        if isinstance(target, DetectionTarget):
            normalized.append(target)
        else:
            normalized.append(DetectionTarget.from_mapping(target))
    return normalized


def normalize_detection_batch(data: DetectionBatch | dict[str, Any]) -> DetectionBatch:
    if isinstance(data, DetectionBatch):
        return data
    return DetectionBatch.from_mapping(data)


def normalize_detection_output(
    output: DetectionModelOutput | dict[str, Any],
) -> DetectionModelOutput:
    if isinstance(output, DetectionModelOutput):
        return output
    strides = output.get("strides_all")
    feature_shapes = output.get("fmp_sizes_all")
    if strides is None:
        strides = [int(output["stride"])]
    if feature_shapes is None:
        feature_shapes = [tuple(output["fmp_size"])]
    extras = {
        key: value
        for key, value in output.items()
        if key not in {"pred_obj", "pred_box", "stride", "fmp_size", "strides_all", "fmp_sizes_all"}
    }
    return DetectionModelOutput(
        pred_obj=output["pred_obj"],
        pred_box=output["pred_box"],
        strides=[int(stride) for stride in strides],
        feature_shapes=[tuple(shape) for shape in feature_shapes],
        extras=extras,
    )


def normalize_detection_detections(
    detections: DetectionDetections | tuple[Any, Any, Any] | list[Any],
) -> DetectionDetections:
    if isinstance(detections, DetectionDetections):
        return detections
    if not isinstance(detections, (tuple, list)) or len(detections) < 2:
        return DetectionDetections.empty()

    boxes = detections[0]
    scores = detections[1]
    labels = detections[2] if len(detections) > 2 else np.zeros((len(scores),), dtype=np.int64)

    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    return DetectionDetections(
        boxes=boxes.float(),
        scores=scores.float(),
        labels=labels.long(),
    )