"""Parsed-graph detection model implementation."""
from __future__ import annotations

from typing import Any

import torch

from afrip.core import BaseDetector
from afrip.models.registry import DETECTORS, build_postprocessor, build_preprocessor
from afrip.nn import ParsedModel


@DETECTORS.register("DetectionModel")
class DetectionModel(BaseDetector):
    """Parsed-graph dense detector assembled from an Ultralytics-style model config."""

    def __init__(
        self,
        model_cfg: dict[str, Any],
        preprocessor_cfg: dict[str, Any],
        postprocessor_cfg: dict[str, Any],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.preprocessor = build_preprocessor(dict(preprocessor_cfg))
        self.postprocessor = build_postprocessor(dict(postprocessor_cfg))
        parsed_cfg = dict(model_cfg)
        parsed_cfg["nc"] = num_classes
        self.model = ParsedModel(parsed_cfg, num_classes=num_classes)

    def _run_dense_model(self, x: torch.Tensor) -> dict[str, Any]:
        x = self.preprocessor(x)
        return self.model(x)

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._run_dense_model(x)
        if outputs["pred_obj"].shape[0] != 1:
            raise ValueError("DetectionModel inference expects batch size 1")

        boxes = outputs["pred_box"][0]
        scores = outputs["pred_obj"][0].squeeze(-1).sigmoid()
        return self.postprocessor(boxes, scores)

    def forward(self, x: torch.Tensor):
        if self.training:
            return self._run_dense_model(x)
        return self.inference(x)