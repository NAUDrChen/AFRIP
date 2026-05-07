"""YoloAssigner：基于网格中心点的标签分配策略（YOLOv1 风格）。"""
from __future__ import annotations

import torch

from afrip.models.registry import ASSIGNERS


@ASSIGNERS.register("YoloAssigner")
class YoloAssigner:
    """将 GT 框分配到对应网格单元的正样本标签分配器。"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self,
        fmp_size: tuple[int, int],
        stride: int,
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = len(targets)
        fmp_h, fmp_w = fmp_size

        gt_objectness = torch.zeros((bs, fmp_h, fmp_w, 1), dtype=torch.float32)
        gt_classes = torch.zeros((bs, fmp_h, fmp_w, self.num_classes), dtype=torch.float32)
        gt_bboxes = torch.zeros((bs, fmp_h, fmp_w, 4), dtype=torch.float32)

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            tgt_cls = targets_per_image["labels"].to(dtype=torch.long)
            tgt_box = targets_per_image["boxes"].to(dtype=torch.float32)

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                x1, y1, x2, y2 = gt_box.tolist()

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
                    cls_one_hot = torch.zeros((self.num_classes,), dtype=torch.float32)
                    cls_one_hot[int(gt_label.item())] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_one_hot
                    gt_bboxes[batch_index, grid_y, grid_x] = gt_box

        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        return gt_objectness, gt_classes, gt_bboxes