"""Task-aligned label assigner ported for AFRIP dense detectors."""
from __future__ import annotations

import torch

from afrip.models.registry import ASSIGNERS
from afrip.utils.box_ops import bbox_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


@ASSIGNERS.register("TaskAlignedAssigner")
class TaskAlignedAssigner:
    """Assign ground truths to dense points using the task-aligned metric."""

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list[int] | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
    ):
        self.topk = int(topk)
        self.topk2 = int(topk2) if topk2 is not None else int(topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stride = [int(value) for value in (stride or [8, 16, 32])]
        self.eps = float(eps)
        self.bs = 0
        self.n_max_boxes = 0

    @torch.no_grad()
    def __call__(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        strides: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if strides is not None:
            self.stride = [int(value) for value in strides]

        self.bs = int(pd_scores.shape[0])
        self.n_max_boxes = int(gt_bboxes.shape[1])

        if self.n_max_boxes == 0:
            empty_scores = torch.zeros_like(pd_scores)
            empty_boxes = torch.zeros_like(pd_bboxes)
            empty_index = torch.zeros_like(pd_scores[..., 0], dtype=torch.long)
            return (
                torch.full_like(empty_index, self.num_classes),
                empty_boxes,
                empty_scores,
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                empty_index,
            )

        return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    def _forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            anc_points,
            mask_gt,
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos,
            overlaps,
            self.n_max_boxes,
            align_metric,
        )
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            mask_in_gts * mask_gt,
        )
        topk = min(self.topk, align_metric.shape[-1])
        mask_topk = self.select_topk_candidates(
            align_metric,
            topk,
            topk_mask=mask_gt.expand(-1, -1, topk).bool(),
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_anchors = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            (self.bs, self.n_max_boxes, num_anchors),
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            (self.bs, self.n_max_boxes, num_anchors),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        indices = torch.zeros((2, self.bs, self.n_max_boxes), dtype=torch.long, device=pd_scores.device)
        indices[0] = torch.arange(end=self.bs, device=pd_scores.device).view(-1, 1).expand(-1, self.n_max_boxes)
        indices[1] = gt_labels.squeeze(-1).clamp_(0, max(self.num_classes - 1, 0))
        bbox_scores[mask_gt] = pd_scores[indices[0], :, indices[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(
        self,
        metrics: torch.Tensor,
        topk: int,
        topk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_metrics, topk_indices = torch.topk(metrics, topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_indices)
        topk_indices = topk_indices.masked_fill(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_indices.device)
        ones = torch.ones_like(topk_indices[:, :, :1], dtype=torch.int8, device=topk_indices.device)
        for k in range(topk):
            count_tensor.scatter_add_(-1, topk_indices[:, :, k : k + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_index = torch.arange(end=self.bs, device=gt_labels.device, dtype=torch.int64)[..., None]
        flat_index = target_gt_idx + batch_index * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[flat_index]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[flat_index]

        target_labels = target_labels.clamp(min=0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=gt_bboxes.dtype,
            device=gt_bboxes.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        fg_scores_mask = fg_mask[:, :, None].expand(-1, -1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0.0)
        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(
        self,
        xy_centers: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        gt_bboxes_xywh = box_xyxy_to_cxcywh(gt_bboxes)
        min_stride = float(self.stride[0]) if self.stride else 1.0
        fallback_stride = float(self.stride[1]) if len(self.stride) > 1 else min_stride
        wh_mask = gt_bboxes_xywh[..., 2:] < min_stride
        stride_value = torch.tensor(fallback_stride, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device)
        gt_bboxes_xywh[..., 2:] = torch.where((wh_mask * mask_gt).bool(), stride_value, gt_bboxes_xywh[..., 2:])
        gt_bboxes = box_cxcywh_to_xyxy(gt_bboxes_xywh)

        num_anchors = xy_centers.shape[0]
        batch_size, num_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
            batch_size,
            num_boxes,
            num_anchors,
            -1,
        )
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(
        self,
        mask_pos: torch.Tensor,
        overlaps: torch.Tensor,
        n_max_boxes: int,
        align_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos
            topk2 = min(self.topk2, align_metric.shape[-1])
            max_align_idx = torch.topk(align_metric, topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_align_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)

        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos