"""Internal neural network blocks shared across AFRIP models."""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def get_conv2d(
    c1: int,
    c2: int,
    k: int,
    p: int,
    s: int,
    d: int,
    g: int,
    bias: bool = False,
) -> nn.Conv2d:
    return nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)


def get_activation(act_type: str | None) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if act_type == "mish":
        return nn.Mish(inplace=True)
    if act_type == "silu":
        return nn.SiLU(inplace=True)
    if act_type is not None:
        return nn.Identity()
    raise NotImplementedError(f"Activation '{act_type}' not implemented.")


def get_norm(norm_type: str | None, dim: int) -> nn.Module:
    if norm_type == "BN":
        return nn.BatchNorm2d(dim)
    if norm_type == "GN":
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    if norm_type is not None:
        return nn.Identity()
    raise NotImplementedError(f"Normalization '{norm_type}' not implemented.")


class Conv(nn.Module):
    """Conv + norm + activation block with optional depthwise mode."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        p: int = 0,
        s: int = 1,
        d: int = 1,
        act_type: str = "lrelu",
        norm_type: str = "BN",
        depthwise: bool = False,
    ):
        super().__init__()
        self.out_channels = c2
        convs: list[nn.Module] = []
        add_bias = not norm_type

        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _adapt_first_conv_weight(state_dict: dict, in_channels: int) -> dict:
    """Adapt ImageNet conv1 weights to arbitrary input channels."""
    key = "conv1.weight"
    if key not in state_dict:
        return state_dict
    weight = state_dict[key]
    if in_channels == 3:
        return state_dict
    if in_channels == 1:
        weight_new = weight.mean(dim=1, keepdim=True)
    elif in_channels == 2:
        weight_new = weight[:, :2, :, :].clone()
    else:
        extra = in_channels - 3
        rand_extra = torch.randn(weight.size(0), extra, weight.size(2), weight.size(3)) * weight.std()
        weight_new = torch.cat([weight, rand_extra], dim=1)
    state_dict[key] = weight_new
    return state_dict