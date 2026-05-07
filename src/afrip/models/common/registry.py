"""Model registries and builders."""
from __future__ import annotations

from typing import Any

from afrip.core import Registry, build_from_config

BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
COMMON_BLOCKS = Registry("common_blocks")
DETECTORS = Registry("detectors")
MATCHERS = Registry("matchers")
LOSSES = Registry("losses")

_POSTPROCESSOR_CONFIG_KEYS = (
    "conf_thresh",
    "nms_thresh",
    "nms_type",
    "soft_nms_method",
    "soft_nms_sigma",
    "soft_nms_score_thresh",
    "topk",
)


def normalize_detector_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    legacy_post_keys = [key for key in _POSTPROCESSOR_CONFIG_KEYS if key in normalized]
    post_cfg = normalized.get("postprocessor_cfg")
    if post_cfg is not None:
        post_cfg = dict(post_cfg)
        post_cfg.setdefault("type", "YOLOObjectnessPostprocessor")
    elif legacy_post_keys:
        post_cfg = {"type": "YOLOObjectnessPostprocessor"}
    else:
        return normalized

    for key in legacy_post_keys:
        if key in normalized:
            post_cfg.setdefault(key, normalized.pop(key))

    normalized["postprocessor_cfg"] = post_cfg
    return normalized


def build_backbone(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, BACKBONES, **extra_kwargs)


def build_neck(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, NECKS, **extra_kwargs)


def build_head(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, HEADS, **extra_kwargs)


def build_common_block(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, COMMON_BLOCKS, **extra_kwargs)


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