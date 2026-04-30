"""YOLORTv2 多尺度检测损失。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from afrip.models.registry import LOSSES, build_matcher


@LOSSES.register("YoloRTv2Criterion")
class YoloRTv2Criterion:
    """多尺度 YOLORTv2 损失：objectness + box，支持 center/pointwise/simota 分配。"""

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
        matcher_cfg: dict[str, Any] | None = None,
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
        self.matcher = build_matcher(matcher_cfg) if matcher_cfg is not None else None

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        cx = boxes[:, 0] * img_w
        cy = boxes[:, 1] * img_h
        w = boxes[:, 2] * img_w
        h = boxes[:, 3] * img_h
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

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

        radius = int(self.cfg["assign_radius"])
        use_inbox = bool(self.cfg["assign_use_inbox"])
        area_weight = float(self.cfg["assign_area_weight"])
        force_one = bool(self.cfg["assign_force_one"])
        force_max_radius = int(self.cfg["assign_force_max_radius"])
        force_level = int(self.cfg["assign_force_level"])

        assigned_gt = torch.full((num_points,), -1, dtype=torch.long, device=device)
        assigned_cost = torch.full((num_points,), float("inf"), dtype=torch.float32, device=device)

        x1 = gt_boxes_xyxy[:, 0]
        y1 = gt_boxes_xyxy[:, 1]
        x2 = gt_boxes_xyxy[:, 2]
        y2 = gt_boxes_xyxy[:, 3]
        cx = x1 + 0.5 * (x2 - x1)
        cy = y1 + 0.5 * (y2 - y1)
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        area_norm = (w * h) / float(max(1, img_w * img_h))

        for lvl, (stride_l, (h_l, w_l)) in enumerate(zip(strides_all, fmp_sizes_all)):
            offset = level_offsets[lvl]
            stride_l = int(stride_l)

            for gt_index in range(num_gt):
                gx0 = int((cx[gt_index] / stride_l).floor().clamp(0, w_l - 1).item())
                gy0 = int((cy[gt_index] / stride_l).floor().clamp(0, h_l - 1).item())
                xmin = max(gx0 - radius, 0)
                xmax = min(gx0 + radius, w_l - 1)
                ymin = max(gy0 - radius, 0)
                ymax = min(gy0 + radius, h_l - 1)

                xs = torch.arange(xmin, xmax + 1, device=device)
                ys = torch.arange(ymin, ymax + 1, device=device)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                xx = xx.reshape(-1)
                yy = yy.reshape(-1)
                px = (xx.to(torch.float32) + 0.5) * float(stride_l)
                py = (yy.to(torch.float32) + 0.5) * float(stride_l)

                if use_inbox:
                    mask = (px >= x1[gt_index]) & (px <= x2[gt_index]) & (py >= y1[gt_index]) & (py <= y2[gt_index])
                    if mask.sum() == 0:
                        continue
                    xx, yy, px, py = xx[mask], yy[mask], px[mask], py[mask]

                idx = offset + (yy * w_l + xx)
                dist = (px - cx[gt_index]).abs() / w[gt_index] + (py - cy[gt_index]).abs() / h[gt_index]
                cost = dist + area_weight * area_norm[gt_index]
                better = cost < assigned_cost[idx]
                if better.any():
                    idx_b = idx[better]
                    assigned_cost[idx_b] = cost[better]
                    assigned_gt[idx_b] = gt_index

        if force_one:
            pos = assigned_gt >= 0
            has = torch.zeros((num_gt,), dtype=torch.bool, device=device)
            if pos.any():
                has.scatter_(0, assigned_gt[pos], True)

            lvl = max(0, min(force_level, len(strides_all) - 1))
            stride_l, (h_l, w_l) = int(strides_all[lvl]), fmp_sizes_all[lvl]
            offset = level_offsets[lvl]
            for gt_index in range(num_gt):
                if has[gt_index]:
                    continue
                gx0 = int((cx[gt_index] / stride_l).floor().clamp(0, w_l - 1).item())
                gy0 = int((cy[gt_index] / stride_l).floor().clamp(0, h_l - 1).item())
                best_idx = None
                best_cost = float("inf")
                for rr in range(0, force_max_radius + 1):
                    xmin = max(gx0 - rr, 0)
                    xmax = min(gx0 + rr, w_l - 1)
                    ymin = max(gy0 - rr, 0)
                    ymax = min(gy0 + rr, h_l - 1)
                    xs = torch.arange(xmin, xmax + 1, device=device)
                    ys = torch.arange(ymin, ymax + 1, device=device)
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                    xx = xx.reshape(-1)
                    yy = yy.reshape(-1)
                    px = (xx.to(torch.float32) + 0.5) * float(stride_l)
                    py = (yy.to(torch.float32) + 0.5) * float(stride_l)
                    idx = offset + (yy * w_l + xx)
                    dist = (px - cx[gt_index]).abs() / w[gt_index] + (py - cy[gt_index]).abs() / h[gt_index]
                    cost = dist + area_weight * area_norm[gt_index]
                    unused = assigned_gt[idx] < 0
                    if unused.any():
                        cand_idx = idx[unused]
                        cand_cost = cost[unused]
                    else:
                        cand_idx = idx
                        cand_cost = cost
                    minv, mini = cand_cost.min(dim=0)
                    val = float(minv.item())
                    if val < best_cost:
                        best_cost = val
                        best_idx = int(cand_idx[mini].item())
                    if best_idx is not None and rr >= 1:
                        break
                if best_idx is not None:
                    assigned_gt[best_idx] = gt_index
                    assigned_cost[best_idx] = best_cost

        pos = assigned_gt >= 0
        if pos.any():
            gt_obj[pos] = 1.0
            gt_box[pos] = gt_boxes_xyxy[assigned_gt[pos]]
        return gt_obj, gt_box

    def __call__(self, outputs: dict[str, Any], targets: list[dict], epoch: int = 0) -> dict[str, torch.Tensor]:
        pred_obj = outputs["pred_obj"]
        pred_box = outputs["pred_box"]
        batch_size, num_points, _ = pred_obj.shape

        strides_all = outputs.get("strides_all")
        fmp_sizes_all = outputs.get("fmp_sizes_all")
        if strides_all is None or fmp_sizes_all is None:
            strides_all = [outputs["stride"]]
            fmp_sizes_all = [outputs["fmp_size"]]

        img_h = max(int(h * s) for (h, _), s in zip(fmp_sizes_all, strides_all))
        img_w = max(int(w * s) for (_, w), s in zip(fmp_sizes_all, strides_all))

        num_level_anchors = [int(h * w) for (h, w) in fmp_sizes_all]
        if sum(num_level_anchors) != num_points:
            raise ValueError(f"pred_obj M={num_points} mismatches multi-scale anchors {num_level_anchors}")

        level_offsets = []
        cur = 0
        for n in num_level_anchors:
            level_offsets.append(cur)
            cur += n

        total_pos = 0.0
        sum_loss_obj = pred_obj.new_tensor(0.0)
        sum_loss_box = pred_box.new_tensor(0.0)
        any_empty = True

        for batch_index in range(batch_size):
            target = targets[batch_index]
            gt_boxes_norm = target["boxes"].to(pred_box.device)
            pred_obj_b = pred_obj[batch_index].view(-1)
            pred_box_b = pred_box[batch_index]
            gt_obj = pred_obj_b.new_zeros((num_points,), dtype=torch.float32)
            gt_box = pred_box_b.new_zeros((num_points, 4), dtype=torch.float32)

            if gt_boxes_norm.shape[0] > 0:
                any_empty = False
                gt_boxes_xyxy = self._cxcywh_to_xyxy(gt_boxes_norm, img_w, img_h)

                if self.assigner == "center":
                    for lvl, (stride_l, (h_l, w_l)) in enumerate(zip(strides_all, fmp_sizes_all)):
                        offset = level_offsets[lvl]
                        cx = gt_boxes_xyxy[:, 0] + 0.5 * (gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0])
                        cy = gt_boxes_xyxy[:, 1] + 0.5 * (gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1])
                        gx = (cx / stride_l).long().clamp(0, w_l - 1)
                        gy = (cy / stride_l).long().clamp(0, h_l - 1)
                        global_idx = offset + (gy * w_l + gx)
                        gt_obj[global_idx] = 1.0
                        gt_box[global_idx] = gt_boxes_xyxy
                elif self.assigner == "pointwise":
                    gt_obj, gt_box = self._assign_targets_pointwise(
                        gt_boxes_xyxy=gt_boxes_xyxy,
                        strides_all=strides_all,
                        fmp_sizes_all=fmp_sizes_all,
                        level_offsets=level_offsets,
                        img_w=img_w,
                        img_h=img_h,
                        num_points=num_points,
                    )
                elif self.assigner == "simota":
                    if self.matcher is None:
                        raise RuntimeError("SimOTA matcher requires matcher_cfg with type=SimOTAMatcher")
                    assigned_gt = self.matcher(
                        pred_obj_logits=pred_obj_b,
                        pred_boxes_xyxy=pred_box_b,
                        gt_boxes_xyxy=gt_boxes_xyxy,
                        strides_all=strides_all,
                        fmp_sizes_all=fmp_sizes_all,
                        level_offsets=level_offsets,
                    )
                    pos = assigned_gt >= 0
                    if pos.any():
                        gt_obj[pos] = 1.0
                        gt_box[pos] = gt_boxes_xyxy[assigned_gt[pos]]
                else:
                    raise ValueError(f"Unknown assigner: {self.assigner}")

                pos_mask = gt_obj > 0.5
                num_pos = int(pos_mask.sum().item())
                total_pos += float(num_pos)
                loss_obj_b = self.loss_objectness(pred_obj_b, gt_obj).sum()
                if num_pos > 0:
                    ious = self._diag_iou(pred_box_b[pos_mask], gt_box[pos_mask])
                    loss_box_b = (1.0 - ious).sum()
                else:
                    loss_box_b = pred_box_b.new_tensor(0.0)
            else:
                loss_obj_b = self.loss_objectness(pred_obj_b, gt_obj).sum() * self.loss_obj_empty_factor
                loss_box_b = pred_box_b.new_tensor(0.0)

            sum_loss_obj += loss_obj_b
            sum_loss_box += loss_box_b

        total_pos = max(total_pos, 1.0)
        loss_obj = sum_loss_obj / total_pos
        loss_box = sum_loss_box / total_pos
        losses = self.loss_obj_weight * loss_obj + self.loss_box_weight * loss_box
        return {
            "losses": losses,
            "loss_obj": loss_obj.detach(),
            "loss_box": loss_box.detach(),
            "num_fgs": torch.tensor(total_pos, device=pred_box.device),
            "empty_frame": any_empty,
        }