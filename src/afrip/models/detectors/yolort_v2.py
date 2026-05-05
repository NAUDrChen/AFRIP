"""YOLORTv2 多尺度单阶段目标检测器。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from afrip.core import BaseDetector
from afrip.modules import build_postprocessor, build_preprocessor
from afrip.models.common import Conv
from afrip.models.registry import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register("YOLORTv2")
class YOLORTv2(BaseDetector):
    """YOLORTv2：P2/P3 双尺度 objectness + box 检测器。"""

    def __init__(
        self,
        backbone_cfg: dict,
        neck_cfg: dict,
        head_cfg: dict,
        num_classes: int = 3,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        nms_type: str = "hard",
        soft_nms_method: str = "linear",
        soft_nms_sigma: float = 0.5,
        soft_nms_score_thresh: float = 1e-3,
        topk: int | None = None,
        preprocessor_cfg: dict | None = None,
        postprocessor_cfg: dict | None = None,
        trainable: bool = False,
        deploy: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nms_type = nms_type
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_score_thresh = soft_nms_score_thresh
        self.topk = topk
        self.strides = [8, 16]
        self.deploy = deploy
        self.set_training_behavior(trainable)

        if preprocessor_cfg is None:
            preprocessor_cfg = {"type": "TensorPreprocessor"}
        self.preprocessor = build_preprocessor(preprocessor_cfg)

        if postprocessor_cfg is None:
            postprocessor_cfg = {
                "type": "YOLOObjectnessPostprocessor",
                "conf_thresh": conf_thresh,
                "nms_thresh": nms_thresh,
                "nms_type": nms_type,
                "soft_nms_method": soft_nms_method,
                "soft_nms_sigma": soft_nms_sigma,
                "soft_nms_score_thresh": soft_nms_score_thresh,
                "topk": topk,
                "class_agnostic": True,
                "num_classes": 1,
            }
        self.postprocessor = build_postprocessor(postprocessor_cfg)

        self.backbone = build_backbone(backbone_cfg)
        self.neck = build_neck(neck_cfg)
        head_dim = self.neck.out_dim

        merged_head_cfg = {**head_cfg, "in_dim": head_dim, "out_dim": head_dim, "num_classes": num_classes}
        self.head_p3 = build_head(merged_head_cfg)
        self.p2_lateral = Conv(128, head_dim, k=1, p=0, s=1, act_type=neck_cfg.get("act_type", "lrelu"), norm_type=neck_cfg.get("norm_type", "BN"))
        self.p2_fuse = Conv(head_dim * 2, head_dim, k=3, p=1, s=1, act_type=neck_cfg.get("act_type", "lrelu"), norm_type=neck_cfg.get("norm_type", "BN"))
        self.head_p2 = build_head(merged_head_cfg)

        self.obj_pred_p3 = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.reg_pred_p3 = nn.Conv2d(head_dim, 4, kernel_size=1)
        self.obj_pred_p2 = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.reg_pred_p2 = nn.Conv2d(head_dim, 4, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)

        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for obj_layer in (self.obj_pred_p2, self.obj_pred_p3):
            assert obj_layer.bias is not None
            bias = obj_layer.bias.view(1, -1)
            bias.data.fill_(bias_value.item())
            obj_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)

        assert self.cls_pred.bias is not None
        bias = self.cls_pred.bias.view(1, -1)
        bias.data.fill_(bias_value.item())
        self.cls_pred.bias = nn.Parameter(bias.view(-1), requires_grad=True)

        for reg_layer in (self.reg_pred_p2, self.reg_pred_p3):
            assert reg_layer.bias is not None
            bias = reg_layer.bias.view(-1)
            bias.data.fill_(1.0)
            reg_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)
            reg_layer.weight = nn.Parameter(torch.zeros_like(reg_layer.weight), requires_grad=True)

    def create_grid(self, fmp_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        hs, ws = fmp_size
        grid_y, grid_x = torch.meshgrid(torch.arange(hs), torch.arange(ws), indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2).to(device)

    def decode_boxes(self, pred_reg: torch.Tensor, fmp_size: tuple[int, int], stride: int) -> torch.Tensor:
        grid_cell = self.create_grid(fmp_size, pred_reg.device)
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * stride
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * stride
        pred_box = torch.cat([pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5], dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def _forward_features(self, x: torch.Tensor):
        c3, c4 = self.backbone(x)
        p3 = self.neck(c4)
        cls_feat_p3, reg_feat_p3 = self.head_p3(p3)
        obj_p3 = self.obj_pred_p3(cls_feat_p3)
        reg_p3 = self.reg_pred_p3(reg_feat_p3)
        fmp3_size = obj_p3.shape[-2:]

        p2_lat = self.p2_lateral(c3)
        p3_up = F.interpolate(p3, size=p2_lat.shape[-2:], mode="nearest")
        p2 = self.p2_fuse(torch.cat([p2_lat, p3_up], dim=1))
        cls_feat_p2, reg_feat_p2 = self.head_p2(p2)
        obj_p2 = self.obj_pred_p2(cls_feat_p2)
        reg_p2 = self.reg_pred_p2(reg_feat_p2)
        fmp2_size = obj_p2.shape[-2:]
        return (obj_p2, reg_p2, fmp2_size), (obj_p3, reg_p3, fmp3_size)

    @torch.no_grad()
    def inference(self, x: torch.Tensor):
        x = self.preprocessor(x)
        (obj_p2, reg_p2, fmp2_size), (obj_p3, reg_p3, fmp3_size) = self._forward_features(x)
        obj_p2 = obj_p2.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_p2 = reg_p2.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        box_p2 = self.decode_boxes(reg_p2, fmp2_size, stride=8)

        obj_p3 = obj_p3.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_p3 = reg_p3.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        box_p3 = self.decode_boxes(reg_p3, fmp3_size, stride=16)

        obj_all = torch.cat([obj_p2, obj_p3], dim=1)[0].squeeze(-1)
        box_all = torch.cat([box_p2, box_p3], dim=1)[0]
        obj_scores = obj_all.sigmoid()

        if self.deploy:
            return torch.cat([box_all, obj_scores[:, None]], dim=-1)

        bboxes, scores, labels = self.postprocessor(box_all.cpu().numpy(), obj_scores.cpu().numpy())
        return bboxes, scores, labels

    def forward(self, x: torch.Tensor):
        if not self.training_behavior_enabled:
            return self.inference(x)

        x = self.preprocessor(x)
        (obj_p2, reg_p2, fmp2_size), (obj_p3, reg_p3, fmp3_size) = self._forward_features(x)
        obj_p2 = obj_p2.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_p2 = reg_p2.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        box_p2 = self.decode_boxes(reg_p2, fmp2_size, stride=8)

        obj_p3 = obj_p3.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_p3 = reg_p3.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        box_p3 = self.decode_boxes(reg_p3, fmp3_size, stride=16)

        return {
            "pred_obj": torch.cat([obj_p2, obj_p3], dim=1),
            "pred_box": torch.cat([box_p2, box_p3], dim=1),
            "stride": 16,
            "fmp_size": fmp3_size,
            "strides_all": self.strides,
            "fmp_sizes_all": [fmp2_size, fmp3_size],
        }