"""Single-scale dense detection loss (BCE objectness + GIoU box)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from afrip.models.detection.assigners.yolo_assigner import YoloAssigner
from afrip.models.registry import LOSSES, build_assigner
from afrip.utils.box_ops import get_ious


@LOSSES.register("YoloRTCriterion")
class YoloRTCriterion:
    """Single-scale dense objectness criterion."""

    def __init__(
        self,
        num_classes: int,
        loss_obj_weight: float = 1.0,
        loss_box_weight: float = 5.0,
        loss_obj_empty_factor: float = 0.25,
        assigner_cfg: dict[str, Any] | None = None,
    ):
        self.num_classes            = num_classes
        self.loss_obj_weight        = loss_obj_weight
        self.loss_box_weight        = loss_box_weight
        self.loss_obj_empty_factor  = loss_obj_empty_factor

        if assigner_cfg is not None:
            self.assigne_impl = build_assigner(assigner_cfg)
        else:
            self.assigner_impl = YoloAssigner(num_classes=num_classes)

    def loss_objectness(self, pred_obj: torch.Tensor,
                        gt_obj: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

    def loss_bboxes(self, pred_box: torch.Tensor,
                    gt_box: torch.Tensor) -> torch.Tensor:
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        return 1.0 - ious

    def __call__(
        self,
        outputs: Any,
        targets: list[dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> dict[str, Any]:
        device = outputs["pred_obj"].device
        stride = int(outputs["stride"])
        fmp_size = tuple(outputs["fmp_size"])
        pred_obj = outputs["pred_obj"]
        pred_box = outputs["pred_box"]
        bs = pred_obj.shape[0]
        gt_obj, _, gt_box = self.assigner_impl(fmp_size, stride, targets)
        gt_obj = gt_obj.to(device)
        gt_box = gt_box.to(device)

        pos_mask = gt_obj.squeeze(-1) > 0
        empty_frame = not bool(pos_mask.any())

        loss_obj = self.loss_objectness(pred_obj.squeeze(-1), gt_obj.squeeze(-1)).mean()
        if pos_mask.any():
            matched_pred = pred_box[pos_mask]
            matched_gt = gt_box[pos_mask]
            loss_box = self.loss_bboxes(matched_pred, matched_gt).mean()
        else:
            loss_box = pred_box.sum() * 0.0
            loss_obj = loss_obj * self.loss_obj_empty_factor

        total_loss = self.loss_obj_weight * loss_obj + self.loss_box_weight * loss_box
        return {
            "loss_obj": loss_obj,
            "loss_box": loss_box,
            "losses": total_loss,
            "empty_frame": empty_frame,
            "batch_size": bs,
            "epoch": epoch,
        }