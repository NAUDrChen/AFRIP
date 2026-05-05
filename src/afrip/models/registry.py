from __future__ import annotations

from typing import Any

from afrip.core import Registry, build_from_config

BACKBONES = Registry("backbones")
NECKS     = Registry("necks")
HEADS     = Registry("heads")
DETECTORS = Registry("detectors")
MATCHERS  = Registry("matchers")
LOSSES    = Registry("losses")

_LEGACY_DETECTOR_ALIASES: dict[str, dict[str, Any]] = {
    "YOLORTv1": {"type": "UnifiedDenseDetector", "architecture": "yolort_v1"},
    "YOLORTv2": {"type": "UnifiedDenseDetector", "architecture": "yolort_v2"},
}


def normalize_detector_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    alias = _LEGACY_DETECTOR_ALIASES.get(str(normalized.get("type", "")))
    if alias is None:
        return normalized
    return {**normalized, **alias}


def build_backbone(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, BACKBONES, **extra_kwargs)


def build_neck(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, NECKS, **extra_kwargs)


def build_head(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, HEADS, **extra_kwargs)


def build_detector(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(
        normalize_detector_config(config),
        DETECTORS,
        **extra_kwargs,
    )


def build_matcher(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, MATCHERS, **extra_kwargs)


def build_loss(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, LOSSES, **extra_kwargs)
