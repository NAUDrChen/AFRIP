"""YoloMatcher：基于网格中心点的标签分配策略（YOLOv1 风格）。"""
from __future__ import annotations

import numpy as np
import torch

from afrip.models.registry import MATCHERS


@MATCHERS.register("YoloMatcher")
class YoloMatcher:
    """将 GT 框分配到对应网格单元的正样本标签分配器。

    Args:
        num_classes: 目标类别数。
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self,
        fmp_size: tuple[int, int],
        stride: int,
        targets: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """标签分配。

        Args:
            fmp_size: (fmp_h, fmp_w) 特征图尺寸。
            stride:   网络输出步长。
            targets:  批次标签列表，每个元素为 dict，包含
                      ``'boxes'`` ([N,4] cx/cy/w/h 归一化) 和 ``'labels'`` ([N])。

        Returns:
            gt_objectness: [B, HW, 1]
            gt_classes:    [B, HW, num_classes]
            gt_bboxes:     [B, HW, 4]  (x1y1x2y2，像素坐标)
        """
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        i_h = fmp_h * stride
        i_w = fmp_w * stride

        gt_objectness = np.zeros([bs, fmp_h, fmp_w, 1])
        gt_classes    = np.zeros([bs, fmp_h, fmp_w, self.num_classes])
        gt_bboxes     = np.zeros([bs, fmp_h, fmp_w, 4])

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            tgt_cls = targets_per_image["labels"].numpy()     # [N,]
            tgt_box = targets_per_image["boxes"].numpy()      # [N, 4] cxcywh 归一化

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                cx, cy, bw, bh = gt_box
                x1 = cx * i_w - bw * i_w * 0.5
                y1 = cy * i_h - bh * i_h * 0.5
                x2 = cx * i_w + bw * i_w * 0.5
                y2 = cy * i_h + bh * i_h * 0.5

                xc = (x2 + x1) * 0.5
                yc = (y2 + y1) * 0.5
                bw = x2 - x1
                bh = y2 - y1

                if bw < 1.0 or bh < 1.0:
                    continue

                grid_x = int(xc / stride)
                grid_y = int(yc / stride)

                if grid_x < fmp_w and grid_y < fmp_h:
                    gt_objectness[batch_index, grid_y, grid_x] = 1.0
                    cls_one_hot = np.zeros(self.num_classes)
                    cls_one_hot[int(gt_label)] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_one_hot
                    gt_bboxes[batch_index, grid_y, grid_x]  = np.array([x1, y1, x2, y2])

        gt_objectness = torch.from_numpy(gt_objectness.reshape(bs, -1, 1)).float()
        gt_classes    = torch.from_numpy(gt_classes.reshape(bs, -1, self.num_classes)).float()
        gt_bboxes     = torch.from_numpy(gt_bboxes.reshape(bs, -1, 4)).float()

        return gt_objectness, gt_classes, gt_bboxes
