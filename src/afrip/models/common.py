"""内部积木模块：卷积、激活、归一化、残差块等，不对外注册，仅供本包内部使用。"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────── 基础卷积工厂 ───────────────────────────

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


def get_conv2d(c1: int, c2: int, k: int, p: int, s: int,
               d: int, g: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)


# ─────────────────────────── 激活 / 归一化工厂 ───────────────────────────

def get_activation(act_type: str | None) -> nn.Module:
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError(f'Activation "{act_type}" not implemented.')


def get_norm(norm_type: str | None, dim: int) -> nn.Module:
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError(f'Normalization "{norm_type}" not implemented.')


# ─────────────────────────── 通用 Conv 模块 ───────────────────────────

class Conv(nn.Module):
    """带 BN + 激活的卷积模块，支持 depthwise 模式。"""

    def __init__(self, c1: int, c2: int, k: int = 1, p: int = 0, s: int = 1,
                 d: int = 1, act_type: str = 'lrelu', norm_type: str = 'BN',
                 depthwise: bool = False):
        super().__init__()
        convs: list[nn.Module] = []
        add_bias = not norm_type

        if depthwise:
            # depthwise + pointwise
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


# ─────────────────────────── 残差块 ───────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1      = conv1x1(inplanes, planes)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = conv3x3(planes, planes, stride)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = conv1x1(planes, planes * self.expansion)
        self.bn3        = nn.BatchNorm2d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─────────────────────────── 预训练权重适配 ───────────────────────────

def _adapt_first_conv_weight(state_dict: dict, in_channels: int) -> dict:
    """将 ImageNet 预训练的 3 通道 conv1 权重适配为任意 in_channels。"""
    key = 'conv1.weight'
    if key not in state_dict:
        return state_dict
    w = state_dict[key]  # [64, 3, 7, 7]
    if in_channels == 3:
        return state_dict
    elif in_channels == 1:
        w_new = w.mean(dim=1, keepdim=True)
    elif in_channels == 2:
        w_new = w[:, :2, :, :].clone()
    else:
        extra = in_channels - 3
        rand_extra = torch.randn(w.size(0), extra, w.size(2), w.size(3)) * w.std()
        w_new = torch.cat([w, rand_extra], dim=1)
    state_dict[key] = w_new
    return state_dict
