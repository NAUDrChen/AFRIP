"""Model-domain registries and builders."""
from __future__ import annotations

from typing import Any

from afrip.core import Registry, build_from_config

BLOCKS = Registry("blocks")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
DETECTORS = Registry("detectors")
ASSIGNERS = Registry("assigners")
LOSSES = Registry("losses")
PREPROCESSORS = Registry("preprocessors")
POSTPROCESSORS = Registry("postprocessors")
TRACKERS = Registry("trackers")


def build_block(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, BLOCKS, **extra_kwargs)


def build_backbone(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, BACKBONES, **extra_kwargs)


def build_neck(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, NECKS, **extra_kwargs)


def build_head(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, HEADS, **extra_kwargs)


def build_detector(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, DETECTORS, **extra_kwargs)


def build_assigner(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, ASSIGNERS, **extra_kwargs)


def build_loss(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, LOSSES, **extra_kwargs)


def build_preprocessor(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, PREPROCESSORS, **extra_kwargs)


def build_postprocessor(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, POSTPROCESSORS, **extra_kwargs)


def build_tracker(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, TRACKERS, **extra_kwargs)