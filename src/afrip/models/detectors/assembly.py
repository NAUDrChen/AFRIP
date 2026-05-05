"""Unified dense detector assembly module."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from afrip.models.detectors.dense_base import DenseDetectorBase
from afrip.models.detectors.specs import build_dense_spec
from afrip.models.registry import DETECTORS


@DETECTORS.register("UnifiedDenseDetector")
class UnifiedDenseDetector(DenseDetectorBase):
    """Config-driven dense detector assembler.

    The model-specific differences are delegated to architecture specs, while
    preprocessing, decode, training outputs and inference flow stay unified.
    """

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        neck_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        architecture: str = "single_scale",
        num_classes: int = 3,
        stride: int = 16,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        nms_type: str = "hard",
        soft_nms_method: str = "linear",
        soft_nms_sigma: float = 0.5,
        soft_nms_score_thresh: float = 1e-3,
        topk: int | None = None,
        preprocessor_cfg: dict[str, Any] | None = None,
        postprocessor_cfg: dict[str, Any] | None = None,
        trainable: bool = False,
        deploy: bool = False,
    ) -> None:
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
        super().__init__(
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
            preprocessor_cfg=preprocessor_cfg,
            postprocessor_cfg=postprocessor_cfg,
            trainable=trainable,
            deploy=deploy,
        )
        self.num_classes = num_classes
        self.stride = stride
        self.architecture = architecture
        self.spec = build_dense_spec(
            architecture=architecture,
            backbone_cfg=backbone_cfg,
            neck_cfg=neck_cfg,
            head_cfg=head_cfg,
            num_classes=num_classes,
        )
        self.strides = list(self.spec.strides)
        self.obj_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        for head in self.spec.heads:
            self.obj_preds.append(nn.Conv2d(head.cls_out_dim, 1, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(head.reg_out_dim, 4, kernel_size=1))
        self._init_predictors()

    def _init_predictors(self) -> None:
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for obj_layer in self.obj_preds:
            if obj_layer.bias is None:
                continue
            bias = obj_layer.bias.view(1, -1)
            bias.data.fill_(bias_value.item())
            obj_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)

        for reg_layer in self.reg_preds:
            if reg_layer.bias is None:
                continue
            bias = reg_layer.bias.view(-1)
            bias.data.fill_(1.0)
            reg_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)
            reg_layer.weight = nn.Parameter(
                torch.zeros_like(reg_layer.weight),
                requires_grad=True,
            )

    def _forward_levels(
        self,
        x: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        outputs = []
        for (cls_feat, reg_feat, stride), obj_pred, reg_pred in zip(
            self.spec(x),
            self.obj_preds,
            self.reg_preds,
        ):
            outputs.append((obj_pred(cls_feat), reg_pred(reg_feat), stride))
        return outputs