"""Reusable neural-network primitives for AFRIP models."""

from afrip.models.registry import BLOCKS, build_block

from .blocks import (
    Conv,
    ConvBlock,
    BasicBlock,
    Bottleneck,
    conv1x1,
    conv3x3,
    get_activation,
    get_norm,
    _adapt_first_conv_weight,
)

__all__ = [
    "BLOCKS",
    "build_block",
    "Conv",
    "ConvBlock",
    "BasicBlock",
    "Bottleneck",
    "conv1x1",
    "conv3x3",
    "get_activation",
    "get_norm",
    "_adapt_first_conv_weight",
]