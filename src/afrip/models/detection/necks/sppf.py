"""SPPF (Spatial Pyramid Pooling - Fast) neck，参考 YOLOv5 实现。"""
from __future__ import annotations

import torch
import torch.nn as nn

from afrip.models.blocks import Conv
from afrip.models.registry import NECKS


@NECKS.register("SPPF")
class SPPF(nn.Module):
    """将骨干输出经空间金字塔池化后升维。"""

    def __init__(self, in_dim: int, out_dim: int, expand_ratio: float = 0.5,
                 pooling_size: int = 5, act_type: str = 'lrelu',
                 norm_type: str = 'BN'):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m   = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))