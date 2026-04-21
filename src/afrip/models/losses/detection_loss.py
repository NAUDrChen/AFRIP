"""YOLORTv1 检测损失（BCE objectness + GIoU box）。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from afrip.models.registry import LOSSES, MATCHERS, build_matcher
from afrip.models.matchers.yolo_matcher import YoloMatcher
from afrip.utils.box_ops import get_ious


@LOSSES.register("YoloRTCriterion")
class YoloRTCriterion:
    """YOLORTv1 损失函数。

    Args:
        num_classes:           目标类别数。
        loss_obj_weight:       objectness 损失权重，默认 1.0。
        loss_box_weight:       box 损失权重，默认 5.0。
        loss_obj_empty_factor: 空帧（无 GT）时 obj 损失的缩放系数，默认 0.25。
        matcher_cfg:           Matcher 配置字典；若为 None 则直接使用 YoloMatcher。
    """

    def __init__(
        self,
        num_classes: int,
        loss_obj_weight: float = 1.0,
        loss_box_weight: float = 5.0,
        loss_obj_empty_factor: float = 0.25,
        matcher_cfg: dict[str, Any] | None = None,
    ):
        self.num_classes            = num_classes
        self.loss_obj_weight        = loss_obj_weight
        self.loss_box_weight        = loss_box_weight
        self.loss_obj_empty_factor  = loss_obj_empty_factor

        if matcher_cfg is not None:
            self.matcher = build_matcher(matcher_cfg)
        else:
            self.matcher = YoloMatcher(num_classes=num_classes)

    # ─────────── 单项损失 ───────────

    def loss_objectness(self, pred_obj: torch.Tensor,
                        gt_obj: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

    def loss_bboxes(self, pred_box: torch.Tensor,
                    gt_box: torch.Tensor) -> torch.Tensor:
        ious = get_ious(pred_box, gt_box, box_mode="xyxy", iou_type='giou')
        return 1.0 - ious

    # ─────────── 前向 ───────────

    def __call__(
        self,
        outputs: dict[str, Any],
        targets: list[dict],
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """计算总损失。

        Args:
            outputs: 模型输出字典，需包含 ``pred_obj`` [B,HW,1]，
                     ``pred_box`` [B,HW,4]，``stride``，``fmp_size``。
            targets: 批次 GT 标签列表。
            epoch:   当前 epoch（保留接口，暂未使用）。

        Returns:
            dict 包含 ``loss_obj``、``loss_box``、``losses``、``empty_frame``。
        """
        device   = outputs['pred_obj'].device
        stride   = outputs['stride']
        fmp_size = outputs['fmp_size']

        pred_obj = outputs['pred_obj'].view(-1)       # [BM,]
        pred_box = outputs['pred_box'].view(-1, 4)    # [BM, 4]

        gt_objectness, _, gt_bboxes = self.matcher(
            fmp_size=fmp_size, stride=stride, targets=targets
        )
        gt_objectness = gt_objectness.view(-1).to(device).float()
        gt_bboxes     = gt_bboxes.view(-1, 4).to(device).float()

        pos_masks = gt_objectness > 0
        num_fgs   = pos_masks.sum()

        if num_fgs.item() == 0:
            loss_obj = (self.loss_objectness(pred_obj, gt_objectness).mean()
                        * self.loss_obj_empty_factor)
            loss_box = pred_obj.new_tensor(0.0)
            return {
                'loss_obj':    loss_obj.detach(),
                'loss_box':    loss_box.detach(),
                'losses':      self.loss_obj_weight * loss_obj,
                'empty_frame': True,
            }

        num_fgs  = num_fgs.clamp(min=1).float()
        loss_obj = self.loss_objectness(pred_obj, gt_objectness).sum() / num_fgs
        loss_box = self.loss_bboxes(pred_box[pos_masks],
                                    gt_bboxes[pos_masks]).sum() / num_fgs

        losses = (self.loss_obj_weight * loss_obj +
                  self.loss_box_weight * loss_box)

        return {
            'loss_obj':    loss_obj,
            'loss_box':    loss_box,
            'losses':      losses,
            'empty_frame': False,
        }
