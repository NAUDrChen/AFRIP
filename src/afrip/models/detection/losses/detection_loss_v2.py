"""Multi-scale dense detection loss."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from afrip.models.registry import LOSSES, build_assigner


@LOSSES.register("YoloRTv2Criterion")
class YoloRTv2Criterion:
    """多尺度 dense objectness 损失：objectness + box，支持 center/pointwise/simota 分配。"""

    def __init__(
        self,
        num_classes: int = 1,
        loss_obj_weight: float = 1.0,
        loss_box_weight: float = 5.0,
        loss_obj_empty_factor: float = 0.25,
        assigner: str = "pointwise",
        assign_radius: int = 1,
        assign_use_inbox: bool = False,
        assign_area_weight: float = 0.05,
        assign_force_one: bool = True,
        assign_force_max_radius: int = 6,
        assign_force_level: int = 0,
        assigner_cfg: dict[str, Any] | None = None,
    ):
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_box_weight = loss_box_weight
        self.loss_obj_empty_factor = loss_obj_empty_factor
        self.assigner = assigner
        self.cfg = {
            "assign_radius": assign_radius,
            "assign_use_inbox": assign_use_inbox,
            "assign_area_weight": assign_area_weight,
            "assign_force_one": assign_force_one,
            "assign_force_max_radius": assign_force_max_radius,
            "assign_force_level": assign_force_level,
        }
        self.assigner_impl = build_assigner(assigner_cfg) if assigner_cfg is not None else None

    @staticmethod
    def _diag_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0:
            return boxes1.new_zeros((0,))
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        return inter / (area1 + area2 - inter + 1e-7)

    def loss_objectness(self, pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction="none")

    def _assign_targets_pointwise(
        self,
        gt_boxes_xyxy: torch.Tensor,
        strides_all,
        fmp_sizes_all,
        level_offsets,
        img_w: int,
        img_h: int,
        num_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = gt_boxes_xyxy.device
        gt_obj = torch.zeros((num_points,), dtype=torch.float32, device=device)
        gt_box = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        num_gt = gt_boxes_xyxy.shape[0]
        if num_gt == 0:
            return gt_obj, gt_box

        for gt_index in range(num_gt):
            x1, y1, x2, y2 = gt_boxes_xyxy[gt_index]
            if x2 <= x1 or y2 <= y1:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = (x2 - x1) * (y2 - y1)
            best_idx = None
            best_cost = None

            for level_index, (stride, fmp_size) in enumerate(zip(strides_all, fmp_sizes_all)):
                h, w = int(fmp_size[0]), int(fmp_size[1])
                offset = int(level_offsets[level_index])
                gx = cx / float(stride) - 0.5
                gy = cy / float(stride) - 0.5
                ix = int(torch.clamp(torch.tensor(gx), min=0, max=max(w - 1, 0)).item())
                iy = int(torch.clamp(torch.tensor(gy), min=0, max=max(h - 1, 0)).item())
                point_index = offset + iy * w + ix
                point_x = (ix + 0.5) * float(stride)
                point_y = (iy + 0.5) * float(stride)
                dist = abs(point_x - float(cx)) + abs(point_y - float(cy))
                cost = dist + self.cfg["assign_area_weight"] * float(area)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = point_index

            if best_idx is not None:
                gt_obj[best_idx] = 1.0
                gt_box[best_idx] = gt_boxes_xyxy[gt_index]

        return gt_obj, gt_box

    def __call__(
        self,
        outputs: Any,
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, Any]:
        pred_obj = outputs["pred_obj"]
        pred_box = outputs["pred_box"]
        strides_all = outputs["strides_all"]
        fmp_sizes_all = outputs["fmp_sizes_all"]
        device = pred_obj.device
        batch_size, num_points, _ = pred_obj.shape

        level_offsets = []
        offset = 0
        for h, w in fmp_sizes_all:
            level_offsets.append(offset)
            offset += int(h * w)

        gt_obj = torch.zeros((batch_size, num_points), dtype=torch.float32, device=device)
        gt_box = torch.zeros((batch_size, num_points, 4), dtype=torch.float32, device=device)

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if self.assigner_impl is not None:
                assigned_gt = self.assigner_impl(
                    pred_obj[batch_index].squeeze(-1),
                    pred_box[batch_index],
                    boxes,
                    strides_all,
                    fmp_sizes_all,
                    level_offsets=level_offsets,
                )
                pos = assigned_gt >= 0
                if pos.any():
                    gt_obj[batch_index, pos] = 1.0
                    gt_box[batch_index, pos] = boxes[assigned_gt[pos]]
            else:
                assigned_obj, assigned_box = self._assign_targets_pointwise(
                    boxes,
                    strides_all,
                    fmp_sizes_all,
                    level_offsets,
                    img_w=0,
                    img_h=0,
                    num_points=num_points,
                )
                gt_obj[batch_index] = assigned_obj
                gt_box[batch_index] = assigned_box

        obj_loss = self.loss_objectness(pred_obj.squeeze(-1), gt_obj).mean()
        pos_mask = gt_obj > 0
        empty_frame = not bool(pos_mask.any())
        if pos_mask.any():
            box_loss = (1.0 - self._diag_iou(pred_box[pos_mask], gt_box[pos_mask])).mean()
        else:
            box_loss = pred_box.sum() * 0.0
            obj_loss = obj_loss * self.loss_obj_empty_factor

        total_loss = self.loss_obj_weight * obj_loss + self.loss_box_weight * box_loss
        return {
            "loss_obj": obj_loss,
            "loss_box": box_loss,
            "losses": total_loss,
            "empty_frame": empty_frame,
            "epoch": epoch,
        }