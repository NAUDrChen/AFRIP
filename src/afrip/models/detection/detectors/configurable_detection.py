"""Config-driven detection model assembly and detector implementation."""
from __future__ import annotations

from typing import Any

import torch

from afrip.core import BaseDetector
from afrip.models.registry import (
    DETECTORS,
    build_backbone,
    build_head,
    build_neck,
    build_postprocessor,
    build_preprocessor,
)


@DETECTORS.register("ConfigurableDetectionModel")
class ConfigurableDetectionModel(BaseDetector):
    """Backbone-neck-head dense detector assembled from registered modules."""

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        neck_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        preprocessor_cfg: dict[str, Any],
        postprocessor_cfg: dict[str, Any],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.preprocessor = build_preprocessor(dict(preprocessor_cfg))
        self.postprocessor = build_postprocessor(dict(postprocessor_cfg))

        self.backbone = build_backbone(backbone_cfg)
        self.neck = build_neck(neck_cfg)
        self.head = build_head(head_cfg, num_classes=num_classes)

    def _run_dense_model(self, x: torch.Tensor) -> dict[str, Any]:
        x = self.preprocessor(x)
        backbone_outputs = self.backbone(x)
        features = self.neck(backbone_outputs)
        return self.head(features)

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._run_dense_model(x)
        if outputs["pred_obj"].shape[0] != 1:
            raise ValueError("ConfigurableDetectionModel inference expects batch size 1")

        boxes = outputs["pred_box"][0]
        scores = outputs["pred_obj"][0].squeeze(-1).sigmoid()
        return self.postprocessor(boxes, scores)

    def forward(self, x: torch.Tensor):
        if self.training:
            return self._run_dense_model(x)
        return self.inference(x)