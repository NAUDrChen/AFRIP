"""Detection-domain builder helpers."""
from __future__ import annotations

from typing import Any

from afrip.models.registry import build_detector, build_loss


def assemble_detection_components(
    cfg: dict[str, Any],
) -> tuple[Any, Any]:
    detector_cfg = dict(cfg["detector"])
    detector = build_detector(detector_cfg)

    loss_cfg = dict(cfg["loss"])
    if "num_classes" not in loss_cfg:
        loss_cfg["num_classes"] = detector_cfg.get("num_classes", 1)
    criterion = build_loss(loss_cfg)
    return detector, criterion