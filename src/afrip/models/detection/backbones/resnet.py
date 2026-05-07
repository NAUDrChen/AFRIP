"""ResNet backbones for AFRIP dense detection models."""
from __future__ import annotations

import torch
import torch.nn as nn

from afrip.models.blocks import BasicBlock, Bottleneck, conv1x1, _adapt_first_conv_weight
from afrip.models.registry import BACKBONES


class ResNet(nn.Module):
    """通用 ResNet 骨干，可配置块类型和层数。"""

    def __init__(self, block: type, layers: list[int],
                 in_channels: int = 3, zero_init_residual: bool = False):
        super().__init__()
        self.inplanes = 64
        self.conv1   = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block, 64,  layers[0])
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: type, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer_list = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(self.inplanes, planes))
        return nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, C, H, W]，输出 [B, 256, H/16, W/16]。"""
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.maxpool(c1)
        c2 = self.layer1(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return c4


@BACKBONES.register("ResNet18")
class ResNet18(ResNet):
    """ResNet-18，输出通道 256，适配单通道雷达输入。"""

    out_channels: int = 256

    def __init__(self, in_channels: int = 1,
                 pretrained: bool = False,
                 pretrained_path: str | None = None):
        super().__init__(BasicBlock, [2, 2, 2], in_channels=in_channels)
        if pretrained and pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict = _adapt_first_conv_weight(state_dict, in_channels)
            self.load_state_dict(state_dict, strict=False)


@BACKBONES.register("ResNet18Pyramid")
class ResNet18Pyramid(nn.Module):
    """ResNet-18 backbone that exposes stride-8 and stride-16 features."""

    out_channels: tuple[int, int] = (128, 256)

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = False,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inplanes = 64
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if pretrained and pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = _adapt_first_conv_weight(state_dict, in_channels)
            self.load_state_dict(state_dict, strict=False)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        return c3, c4