"""YOLORTv1 单阶段目标检测器（ResNet18 + SPPF + DecoupledHead）。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from afrip.core import BaseDetector
from afrip.models.registry import (
    DETECTORS, BACKBONES, NECKS, HEADS,
    build_backbone, build_neck, build_head,
)
from afrip.utils.nms import multiclass_nms


@DETECTORS.register("YOLORTv1")
class YOLORTv1(BaseDetector):
    """YOLORTv1 探测器。

    Args:
        backbone_cfg: 骨干网络配置字典（含 ``type``）。
        neck_cfg:     Neck 配置字典（含 ``type``）。
        head_cfg:     Head 配置字典（含 ``type``，不含 in_dim/out_dim/num_classes，
                      由探测器自动注入）。
        num_classes:  类别数，默认 3。
        stride:       输出步长，默认 16。
        conf_thresh:  推理置信度阈值，默认 0.01。
        nms_thresh:   NMS IoU 阈值，默认 0.5。
        trainable:    True 时 forward 返回 dict（用于训练），否则返回后处理结果。
        deploy:       True 时 inference 返回原始张量（用于导出）。
    """

    def __init__(
        self,
        backbone_cfg: dict,
        neck_cfg: dict,
        head_cfg: dict,
        num_classes: int = 3,
        stride: int = 16,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        trainable: bool = False,
        deploy: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride      = stride
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.deploy      = deploy
        self.set_training_behavior(trainable)

        # ── 组件构建 ─────────────────────────────────────────
        self.backbone = build_backbone(backbone_cfg)
        self.neck     = build_neck(neck_cfg)
        head_dim      = self.neck.out_dim

        # 将运行时维度注入 head_cfg（不修改调用方传入的原始 dict）
        merged_head_cfg = {
            **head_cfg,
            "in_dim":      head_dim,
            "out_dim":     head_dim,
            "num_classes": num_classes,
        }
        self.head = build_head(merged_head_cfg)

        # ── 预测卷积层 ────────────────────────────────────────
        cls_out_dim = self.head.cls_out_dim
        reg_out_dim = self.head.reg_out_dim
        self.obj_pred = nn.Conv2d(cls_out_dim, 1,           kernel_size=1)
        self.cls_pred = nn.Conv2d(cls_out_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(reg_out_dim, 4,           kernel_size=1)

        # ── 预测层初始化 ──────────────────────────────────────
        init_prob   = 0.01
        bias_value  = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))

        for pred in (self.obj_pred, self.cls_pred):
            b = pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            pred.bias = nn.Parameter(b.view(-1), requires_grad=True)

        b = self.reg_pred.bias.view(-1)
        b.data.fill_(1.0)
        self.reg_pred.bias   = nn.Parameter(b.view(-1), requires_grad=True)
        self.reg_pred.weight = nn.Parameter(
            torch.zeros_like(self.reg_pred.weight), requires_grad=True
        )

    # ─────────────────────────── 辅助方法 ───────────────────────────

    def create_grid(self, fmp_size: tuple[int, int],
                    device: torch.device) -> torch.Tensor:
        """生成网格坐标矩阵 [HW, 2]，坐标为像素单位。"""
        hs, ws = fmp_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(hs), torch.arange(ws), indexing='ij'
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        return grid_xy.view(-1, 2).to(device)

    def decode_boxes(self, pred_reg: torch.Tensor,
                     fmp_size: tuple[int, int]) -> torch.Tensor:
        """将网络原始输出解码为 x1y1x2y2 像素坐标。

        Args:
            pred_reg:  [B, HW, 4] 或 [HW, 4]，原始回归输出。
            fmp_size:  (H, W) 特征图尺寸。

        Returns:
            pred_box:  与 pred_reg 形状相同，x1y1x2y2 格式。
        """
        grid_cell = self.create_grid(fmp_size, pred_reg.device)
        pred_ctr  = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        wh_log    = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh   = torch.exp(wh_log) * self.stride
        pred_box  = torch.cat(
            [pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5], dim=-1
        )
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def postprocess(
        self,
        bboxes: np.ndarray,
        obj_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """阈值过滤 + class-agnostic NMS。

        Returns:
            (bboxes, scores, labels) after NMS.
        """
        keep        = np.where(obj_scores >= self.conf_thresh)[0]
        bboxes      = bboxes[keep]
        scores      = obj_scores[keep]
        labels      = np.zeros(len(scores), dtype=np.int32)

        if len(bboxes) == 0:
            return bboxes, scores, labels

        finite  = np.isfinite(bboxes).all(axis=1)
        proper  = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        mask    = finite & proper
        bboxes  = bboxes[mask]
        scores  = scores[mask]
        labels  = labels[mask]

        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, 1, class_agnostic=True
        )
        return bboxes, scores, labels

    # ─────────────────────────── 推理 / 训练 ───────────────────────────

    @torch.no_grad()
    def inference(
        self, x: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | torch.Tensor:
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B,HW,1]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B,HW,4]

        # 推理默认 batch=1
        obj_pred    = obj_pred[0].squeeze(-1)   # [HW,]
        reg_pred    = reg_pred[0]               # [HW, 4]
        bboxes      = self.decode_boxes(reg_pred, fmp_size)
        obj_scores  = obj_pred.sigmoid()

        if self.deploy:
            return torch.cat([bboxes, obj_scores[:, None]], dim=-1)

        bboxes     = bboxes.cpu().numpy()
        obj_scores = obj_scores.cpu().numpy()
        return self.postprocess(bboxes, obj_scores)

    def forward(self, x: torch.Tensor):
        if not self.training_behavior_enabled:
            return self.inference(x)

        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)   # [B,HW,1]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)   # [B,HW,4]
        box_pred = self.decode_boxes(reg_pred, fmp_size)

        return {
            "pred_obj": obj_pred,     # [B, HW, 1]
            "pred_box": box_pred,     # [B, HW, 4]
            "stride":   self.stride,
            "fmp_size": fmp_size,
        }
