"""Task-aligned detection loss for yolo26-style AFRIP detectors."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from afrip.models.registry import LOSSES, build_assigner
from afrip.utils.box_ops import bbox2dist, bbox_iou


def make_anchors(
    fmp_sizes_all: list[tuple[int, int]],
    strides_all: list[int],
    device: torch.device,
    dtype: torch.dtype,
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build absolute anchor centers and per-point strides from AFRIP level metadata."""
    anchor_points = []
    stride_tensor = []
    for (height, width), stride in zip(fmp_sizes_all, strides_all):
        ys = torch.arange(height, device=device, dtype=dtype) + grid_cell_offset
        xs = torch.arange(width, device=device, dtype=dtype) + grid_cell_offset
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        points = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2) * float(stride)
        anchor_points.append(points)
        stride_tensor.append(torch.full((height * width, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)


class DFLoss(nn.Module):
    """Distribution focal loss for dense box regression."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left
        return (
            F.cross_entropy(pred_dist, target_left.view(-1), reduction="none").view(target_left.shape) * weight_left
            + F.cross_entropy(pred_dist, target_right.view(-1), reduction="none").view(target_right.shape) * weight_right
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """CIoU + optional DFL loss for dense predictions."""

    def __init__(self, reg_max: int = 1):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is None:
            return loss_iou, pred_dist.sum() * 0.0

        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


@LOSSES.register("Yolo26Criterion")
class Yolo26Criterion:
    """Task-aligned BCE/CIoU/DFL loss compatible with AFRIP DetectionModel outputs."""

    def __init__(
        self,
        num_classes: int = 1,
        loss_cls_weight: float = 0.5,
        loss_box_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        reg_max: int = 1,
        end2end: bool = True,
        total_epochs: int = 1,
        one2many_weight: float = 0.8,
        one2one_weight: float = 0.2,
        one2many_final_weight: float = 0.1,
        assigner_cfg: dict[str, Any] | None = None,
        one2one_assigner_cfg: dict[str, Any] | None = None,
    ):
        self.num_classes = int(num_classes)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_box_weight = float(loss_box_weight)
        self.loss_dfl_weight = float(loss_dfl_weight)
        self.reg_max = int(reg_max)
        self.end2end = bool(end2end)
        self.total_epochs = int(total_epochs)
        self.one2many_weight = float(one2many_weight)
        self.one2one_weight = float(one2one_weight)
        self.one2many_final_weight = float(one2many_final_weight)
        self.total_branch_weight = self.one2many_weight + self.one2one_weight

        base_assigner_cfg = dict(assigner_cfg or {"type": "TaskAlignedAssigner", "topk": 10, "alpha": 0.5, "beta": 6.0})
        base_assigner_cfg.setdefault("num_classes", self.num_classes)
        self.assigner = build_assigner(base_assigner_cfg)

        one2one_cfg = dict(one2one_assigner_cfg or base_assigner_cfg)
        one2one_cfg.setdefault("num_classes", self.num_classes)
        one2one_cfg.setdefault("topk", 7)
        one2one_cfg.setdefault("topk2", 1)
        self.one2one_assigner = build_assigner(one2one_cfg)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max=self.reg_max)

    def _pack_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_gt = max((int(target["boxes"].shape[0]) for target in targets), default=0)
        gt_labels = torch.zeros((batch_size, max_gt, 1), dtype=torch.long, device=device)
        gt_bboxes = torch.zeros((batch_size, max_gt, 4), dtype=torch.float32, device=device)
        mask_gt = torch.zeros((batch_size, max_gt, 1), dtype=torch.bool, device=device)

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            labels = target["labels"].to(device=device, dtype=torch.long)
            count = int(boxes.shape[0])
            if count == 0:
                continue
            gt_labels[batch_index, :count, 0] = labels
            gt_bboxes[batch_index, :count] = boxes
            mask_gt[batch_index, :count, 0] = True

        return gt_labels, gt_bboxes, mask_gt

    def _branch_decay(self, epoch: int) -> tuple[float, float]:
        if not self.end2end:
            return 1.0, 0.0
        if self.total_epochs <= 1:
            return self.one2many_weight, self.one2one_weight

        progress = min(max(int(epoch), 0), self.total_epochs - 1) / max(self.total_epochs - 1, 1)
        o2m = (1.0 - progress) * (self.one2many_weight - self.one2many_final_weight) + self.one2many_final_weight
        o2o = max(self.total_branch_weight - o2m, 0.0)
        return o2m, o2o

    def _loss_single(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        assigner,
    ) -> dict[str, Any]:
        pred_cls = outputs["pred_cls"]
        pred_box = outputs["pred_box"]
        pred_dist = outputs["pred_dist"]
        device = pred_cls.device
        dtype = pred_cls.dtype
        batch_size = int(pred_cls.shape[0])

        anchor_points, _ = make_anchors(outputs["fmp_sizes_all"], outputs["strides_all"], device, dtype)
        gt_labels, gt_bboxes, mask_gt = self._pack_targets(targets, batch_size, device)
        _, target_bboxes, target_scores, fg_mask, _ = assigner(
            pred_cls.detach().sigmoid(),
            pred_box.detach(),
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
            strides=outputs["strides_all"],
        )

        target_scores = target_scores.to(dtype=dtype)
        target_scores_sum = target_scores.sum().clamp(min=1.0)
        loss_cls = self.bce(pred_cls, target_scores).sum() / target_scores_sum

        if fg_mask.any():
            loss_box, loss_dfl = self.bbox_loss(
                pred_dist,
                pred_box,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        else:
            loss_box = pred_box.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        return {
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_dfl": loss_dfl,
            "empty_frame": not bool(fg_mask.any()),
        }

    def __call__(
        self,
        outputs: Any,
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, Any]:
        if self.end2end and isinstance(outputs, dict) and "one2many" in outputs and "one2one" in outputs:
            loss_many = self._loss_single(outputs["one2many"], targets, self.assigner)
            loss_one = self._loss_single(outputs["one2one"], targets, self.one2one_assigner)
            w_many, w_one = self._branch_decay(epoch)
            loss_box = loss_many["loss_box"] * w_many + loss_one["loss_box"] * w_one
            loss_cls = loss_many["loss_cls"] * w_many + loss_one["loss_cls"] * w_one
            loss_dfl = loss_many["loss_dfl"] * w_many + loss_one["loss_dfl"] * w_one
            empty_frame = loss_many["empty_frame"] and loss_one["empty_frame"]
        else:
            loss_single = self._loss_single(outputs, targets, self.assigner)
            loss_box = loss_single["loss_box"]
            loss_cls = loss_single["loss_cls"]
            loss_dfl = loss_single["loss_dfl"]
            empty_frame = loss_single["empty_frame"]

        total_loss = (
            self.loss_box_weight * loss_box
            + self.loss_cls_weight * loss_cls
            + self.loss_dfl_weight * loss_dfl
        )
        return {
            "loss_box": loss_box,
            "loss_cls": loss_cls,
            "loss_dfl": loss_dfl,
            "losses": total_loss,
            "empty_frame": empty_frame,
            "epoch": epoch,
        }