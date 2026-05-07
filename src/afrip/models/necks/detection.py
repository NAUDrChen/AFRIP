"""Registered necks for AFRIP dense detection models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from afrip.models.common import Conv, NECKS

from .sppf import SPPF


def _select_backbone_feature(
    outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    index: int | None,
) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        if index not in (None, 0):
            raise IndexError("Backbone returned a single tensor, but indexed access was requested")
        return outputs
    if index is None:
        if len(outputs) != 1:
            raise ValueError("Backbone returned multiple tensors; neck config must specify an input index")
        return outputs[0]
    return outputs[index]


@NECKS.register("SingleScaleSPPFNeck")
class SingleScaleSPPFNeck(nn.Module):
    """Single-scale SPPF neck that returns one named feature map."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        output_name: str = "p3",
        input_index: int | None = None,
        expand_ratio: float = 0.5,
        pooling_size: int = 5,
        act_type: str = "lrelu",
        norm_type: str = "BN",
    ) -> None:
        super().__init__()
        self.output_name = output_name
        self.input_index = input_index
        self.sppf = SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=expand_ratio,
            pooling_size=pooling_size,
            act_type=act_type,
            norm_type=norm_type,
        )

    def forward(
        self,
        backbone_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        feature = _select_backbone_feature(backbone_outputs, self.input_index)
        return {self.output_name: self.sppf(feature)}


@NECKS.register("PyramidFusionNeck")
class PyramidFusionNeck(nn.Module):
    """Two-scale neck that builds a top-down pyramid from stride-8/16 features."""

    def __init__(
        self,
        low_in_dim: int,
        high_in_dim: int,
        out_dim: int,
        low_output_name: str = "p2",
        high_output_name: str = "p3",
        low_input_index: int = 0,
        high_input_index: int = 1,
        expand_ratio: float = 0.5,
        pooling_size: int = 5,
        act_type: str = "lrelu",
        norm_type: str = "BN",
    ) -> None:
        super().__init__()
        self.low_output_name = low_output_name
        self.high_output_name = high_output_name
        self.low_input_index = low_input_index
        self.high_input_index = high_input_index
        self.high_neck = SPPF(
            in_dim=high_in_dim,
            out_dim=out_dim,
            expand_ratio=expand_ratio,
            pooling_size=pooling_size,
            act_type=act_type,
            norm_type=norm_type,
        )
        self.low_lateral = Conv(
            low_in_dim,
            out_dim,
            k=1,
            p=0,
            s=1,
            act_type=act_type,
            norm_type=norm_type,
        )
        self.low_fuse = Conv(
            out_dim * 2,
            out_dim,
            k=3,
            p=1,
            s=1,
            act_type=act_type,
            norm_type=norm_type,
        )

    def forward(
        self,
        backbone_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        low_feature = _select_backbone_feature(backbone_outputs, self.low_input_index)
        high_feature = _select_backbone_feature(backbone_outputs, self.high_input_index)

        high_out = self.high_neck(high_feature)
        low_lateral = self.low_lateral(low_feature)
        high_up = F.interpolate(high_out, size=low_lateral.shape[-2:], mode="nearest")
        low_out = self.low_fuse(torch.cat([low_lateral, high_up], dim=1))

        return {
            self.low_output_name: low_out,
            self.high_output_name: high_out,
        }