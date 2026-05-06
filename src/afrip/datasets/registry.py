from __future__ import annotations

from typing import Any

from afrip.core import Registry, build_from_config

DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, image, boxes):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        return image, boxes


def build_transform_pipeline(configs: list[dict[str, Any]] | None) -> Any:
    if not configs:
        return None
    transforms = [build_from_config(cfg, TRANSFORMS) for cfg in configs]
    return Compose(transforms)


def build_dataset(config: dict[str, Any], **extra_kwargs: Any) -> Any:
    return build_from_config(config, DATASETS, **extra_kwargs)