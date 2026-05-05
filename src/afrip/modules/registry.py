from __future__ import annotations

from typing import Any

from afrip.core import Registry, build_from_config

PREPROCESSORS = Registry("preprocessors")
POSTPROCESSORS = Registry("postprocessors")


def build_preprocessor(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, PREPROCESSORS, **extra_kwargs)


def build_postprocessor(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, POSTPROCESSORS, **extra_kwargs)