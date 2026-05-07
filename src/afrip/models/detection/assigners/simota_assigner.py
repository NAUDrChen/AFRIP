"""SimOTA 标签分配器，面向多尺度点集动态 K 分配。"""
from __future__ import annotations

import torch

from afrip.models.registry import ASSIGNERS


@ASSIGNERS.register("SimOTAAssigner")
class SimOTAAssigner:
    """YOLOX/SimOTA 风格动态 K 标签分配器，仅使用 objectness + IoU cost。"""

    def __init__(
        self,
        topk: int = 10,
        center_radius: float = 2.5,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
        use_inbox: bool = True,
        use_center: bool = True,
        force_one: bool = True,
        force_level: int = 0,
    ):
        self.topk = topk
        self.center_radius = center_radius
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.use_inbox = use_inbox
        self.use_center = use_center
        self.force_one = force_one
        self.force_level = force_level
        self._cache_key = None
        self._cache_points = None
        self._cache_strides = None

    @staticmethod
    def _bbox_iou_vec(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        x1 = torch.maximum(boxes[:, 0], box[0])
        y1 = torch.maximum(boxes[:, 1], box[1])
        x2 = torch.minimum(boxes[:, 2], box[2])
        y2 = torch.minimum(boxes[:, 3], box[3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        area2 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        union = area1 + area2 - inter + 1e-7
        return inter / union

    def _build_points(self, strides_all, fmp_sizes_all, device: torch.device):
        key = (
            tuple(int(s) for s in strides_all),
            tuple((int(h), int(w)) for (h, w) in fmp_sizes_all),
            str(device),
        )
        if self._cache_key == key and self._cache_points is not None:
            return self._cache_points, self._cache_strides

        points_list = []
        stride_list = []
        for stride, (h, w) in zip(strides_all, fmp_sizes_all):
            ys = torch.arange(h, device=device, dtype=torch.float32)
            xs = torch.arange(w, device=device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            px = (xx.reshape(-1) + 0.5) * float(stride)
            py = (yy.reshape(-1) + 0.5) * float(stride)
            points_list.append(torch.stack([px, py], dim=-1))
            stride_list.append(torch.full((h * w,), float(stride), device=device, dtype=torch.float32))

        points = torch.cat(points_list, dim=0)
        strides = torch.cat(stride_list, dim=0)
        self._cache_key = key
        self._cache_points = points
        self._cache_strides = strides
        return points, strides

    @torch.no_grad()
    def __call__(
        self,
        pred_obj_logits: torch.Tensor,
        pred_boxes_xyxy: torch.Tensor,
        gt_boxes_xyxy: torch.Tensor,
        strides_all,
        fmp_sizes_all,
        level_offsets=None,
    ) -> torch.Tensor:
        device = pred_boxes_xyxy.device
        num_points = pred_boxes_xyxy.shape[0]
        num_gt = gt_boxes_xyxy.shape[0]

        assigned_gt = torch.full((num_points,), -1, dtype=torch.long, device=device)
        if num_gt == 0 or num_points == 0:
            return assigned_gt

        points, strides = self._build_points(strides_all, fmp_sizes_all, device=device)
        x1, y1, x2, y2 = gt_boxes_xyxy[:, 0], gt_boxes_xyxy[:, 1], gt_boxes_xyxy[:, 2], gt_boxes_xyxy[:, 3]
        gcx = x1 + 0.5 * (x2 - x1)
        gcy = y1 + 0.5 * (y2 - y1)
        px, py = points[:, 0], points[:, 1]
        best_cost = torch.full((num_points,), float("inf"), device=device, dtype=torch.float32)

        for gt_index in range(num_gt):
            in_box = (px >= x1[gt_index]) & (px <= x2[gt_index]) & (py >= y1[gt_index]) & (py <= y2[gt_index])
            radius = self.center_radius * strides
            in_center = ((px - gcx[gt_index]).abs() < radius) & ((py - gcy[gt_index]).abs() < radius)

            if self.use_inbox and self.use_center:
                cand_mask = in_box & in_center
            elif self.use_inbox:
                cand_mask = in_box
            elif self.use_center:
                cand_mask = in_center
            else:
                cand_mask = torch.ones_like(in_box)

            cand_idx = torch.where(cand_mask)[0]
            if cand_idx.numel() == 0:
                continue

            ious = self._bbox_iou_vec(pred_boxes_xyxy[cand_idx], gt_boxes_xyxy[gt_index])
            cls_cost = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_obj_logits[cand_idx], torch.ones_like(ious), reduction="none"
            )
            cost = self.cls_weight * cls_cost + self.iou_weight * (1.0 - ious)

            topk = min(self.topk, ious.numel())
            iou_topk, _ = torch.topk(ious, k=topk, largest=True)
            dynamic_k = int(torch.clamp(iou_topk.sum(), min=1.0).item())
            dynamic_k = max(1, min(dynamic_k, cand_idx.numel()))

            _, pos_inds = torch.topk(cost, k=dynamic_k, largest=False)
            sel_idx = cand_idx[pos_inds]
            sel_cost = cost[pos_inds]
            cur_best = best_cost[sel_idx]
            better = sel_cost < cur_best
            if better.any():
                sel_idx = sel_idx[better]
                sel_cost = sel_cost[better]
                best_cost[sel_idx] = sel_cost
                assigned_gt[sel_idx] = gt_index

        if self.force_one:
            pos = assigned_gt >= 0
            has = torch.zeros((num_gt,), dtype=torch.bool, device=device)
            if pos.any():
                has.scatter_(0, assigned_gt[pos], True)

            lvl = max(0, min(self.force_level, len(fmp_sizes_all) - 1))
            h_l, w_l = fmp_sizes_all[lvl]
            offset = int(level_offsets[lvl]) if level_offsets is not None else 0
            start = offset
            end = offset + int(h_l * w_l)
            px_l = px[start:end]
            py_l = py[start:end]

            for gt_index in range(num_gt):
                if has[gt_index]:
                    continue
                dist = (px_l - gcx[gt_index]).abs() + (py_l - gcy[gt_index]).abs()
                idx = start + int(dist.argmin().item())
                assigned_gt[idx] = gt_index

        return assigned_gt