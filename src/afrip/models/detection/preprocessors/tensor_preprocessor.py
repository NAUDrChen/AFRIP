"""Tensor 输入预处理组件。"""
from __future__ import annotations

import torch

from afrip.models.registry import PREPROCESSORS


@PREPROCESSORS.register("TensorPreprocessor")
class TensorPreprocessor:
    """对 detector 输入张量做轻量预处理。"""

    def __init__(
        self,
        scale: float = 1.0,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        if self.scale != 1.0:
            y = y * self.scale

        if self.mean is not None:
            mean = torch.tensor(self.mean, dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
            y = y - mean

        if self.std is not None:
            std = torch.tensor(self.std, dtype=y.dtype, device=y.device).view(1, -1, 1, 1)
            y = y / std.clamp_min(torch.finfo(y.dtype).eps)

        if self.clamp_min is not None or self.clamp_max is not None:
            min_value = self.clamp_min if self.clamp_min is not None else -torch.inf
            max_value = self.clamp_max if self.clamp_max is not None else torch.inf
            y = y.clamp(min=min_value, max=max_value)

        return y