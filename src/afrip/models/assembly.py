"""High-level model assembly helpers."""
from __future__ import annotations

from typing import Any

from afrip.models.registry import build_detector, build_loss, normalize_detector_config


def assemble_detection_components(
    cfg: dict[str, Any],
    trainable: bool | None = None,
) -> tuple[Any, Any]:
    detector_cfg = normalize_detector_config(dict(cfg["detector"]))
    if trainable is not None:
        detector_cfg["trainable"] = trainable

    detector = build_detector(detector_cfg)

    loss_cfg = dict(cfg["loss"])
    if "num_classes" not in loss_cfg:
        loss_cfg["num_classes"] = detector_cfg.get("num_classes", 1)
    criterion = build_loss(loss_cfg)
    return detector, criterion