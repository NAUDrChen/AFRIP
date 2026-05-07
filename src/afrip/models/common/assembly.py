"""Config-driven detection model assembly and high-level build helpers."""
from __future__ import annotations

from typing import Any

import torch

from afrip.core import BaseDetector
from afrip.modules import build_postprocessor, build_preprocessor
from .registry import (
    DETECTORS,
    build_backbone,
    build_detector,
    build_head,
    build_loss,
    build_neck,
)


@DETECTORS.register("ConfigurableDetectionModel")
class ConfigurableDetectionModel(BaseDetector):
    """Backbone-neck-head dense detector assembled from registered modules.

    Interfaces:
    - backbone: returns Tensor or sequence[Tensor]
    - neck: consumes backbone outputs and returns named feature tensors
    - head: consumes named feature tensors and returns a plain output dict
    - training forward returns plain output dict
    - inference returns plain detection dict
    """

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        neck_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        preprocessor_cfg: dict[str, Any],
        postprocessor_cfg: dict[str, Any],
        num_classes: int = 1,
        trainable: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.deploy = deploy
        self.set_training_behavior(trainable)

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
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        outputs = self._run_dense_model(x)
        if outputs["pred_obj"].shape[0] != 1:
            raise ValueError("ConfigurableDetectionModel inference expects batch size 1")

        boxes = outputs["pred_box"][0]
        scores = outputs["pred_obj"][0].squeeze(-1).sigmoid()
        if self.deploy:
            return torch.cat([boxes, scores[:, None]], dim=-1)

        return self.postprocessor(boxes, scores)

    def forward(self, x: torch.Tensor):
        if self.training_behavior_enabled:
            return self._run_dense_model(x)
        return self.inference(x)


def assemble_detection_components(
    cfg: dict[str, Any],
    trainable: bool | None = None,
) -> tuple[Any, Any]:
    detector_cfg = dict(cfg["detector"])
    if trainable is not None:
        detector_cfg["trainable"] = trainable

    detector = build_detector(detector_cfg)

    loss_cfg = dict(cfg["loss"])
    if "num_classes" not in loss_cfg:
        loss_cfg["num_classes"] = detector_cfg.get("num_classes", 1)
    criterion = build_loss(loss_cfg)
    return detector, criterion