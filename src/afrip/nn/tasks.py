"""Ultralytics-style model graph parsing for AFRIP detectors."""
from __future__ import annotations

import ast
import contextlib
import math
from typing import Any

import torch
import torch.nn as nn

from .modules import (
    AIFI,
    C2f,
    Concat,
    Conv,
    Detect,
    DetectContract,
    DetectDecode,
    DWConv,
    Bottleneck,
    ResNetLayer,
    SPPF,
)


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round channel counts up so width scaling keeps divisible tensors."""
    return int(math.ceil(x / divisor) * divisor)


def parse_model(
    config: dict[str, Any],
    ch: int,
    num_classes: int,
    verbose: bool = False,
) -> tuple[nn.Sequential, list[int]]:
    """Parse a YOLO-style graph definition into an executable PyTorch module list."""
    del verbose

    depth = float(config.get("depth_multiple", 1.0))
    width = float(config.get("width_multiple", 1.0))
    layers: list[nn.Module] = []
    save: list[int] = []
    channels = [int(ch)]

    base_modules = frozenset(
        {
            Conv,
            DWConv,
            Bottleneck,
            C2f,
            SPPF,
            ResNetLayer,
        }
    )
    repeat_modules = frozenset({C2f})

    for index, (from_idx, repeats, module_name, args) in enumerate(config["backbone"] + config["head"]):
        module_cls = _resolve_module(module_name)
        layer_args = list(args)
        for arg_index, value in enumerate(layer_args):
            if isinstance(value, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    layer_args[arg_index] = ast.literal_eval(value)

        repeats = max(round(repeats * depth), 1) if repeats > 1 else repeats

        if module_cls in base_modules:
            input_channels = channels[from_idx]
            out_channels = layer_args[0]
            out_channels = make_divisible(out_channels * width, 8)
            layer_args = [input_channels, out_channels, *layer_args[1:]]
            if module_cls in repeat_modules:
                layer_args.insert(2, repeats)
                repeats = 1
            layer_out_channels = out_channels
            if module_cls is ResNetLayer:
                is_first = bool(layer_args[3]) if len(layer_args) > 3 else False
                expansion = int(layer_args[5]) if len(layer_args) > 5 else 4
                layer_out_channels = out_channels if is_first else out_channels * expansion
        elif module_cls is AIFI:
            layer_args = [channels[from_idx], *layer_args]
            layer_out_channels = channels[from_idx]
        elif module_cls is Concat:
            layer_out_channels = sum(channels[layer] for layer in from_idx)
        elif module_cls is Detect:
            feature_indices = from_idx if isinstance(from_idx, list) else [from_idx]
            layer_args = [num_classes, [channels[layer] for layer in feature_indices], *layer_args]
            layer_out_channels = sum(channels[layer] for layer in feature_indices)
        elif module_cls in {DetectDecode, DetectContract}:
            layer_out_channels = channels[from_idx] if isinstance(from_idx, int) else sum(channels[layer] for layer in from_idx)
        elif module_cls in {nn.Upsample, nn.Identity, nn.BatchNorm2d, nn.MaxPool2d}:
            if module_cls is nn.BatchNorm2d:
                layer_args = [channels[from_idx]]
            layer_out_channels = channels[from_idx]
        else:
            raise KeyError(f"Unsupported parsed module: {module_name}")

        module = nn.Sequential(*(module_cls(*layer_args) for _ in range(repeats))) if repeats > 1 else module_cls(*layer_args)
        module.np = sum(param.numel() for param in module.parameters())
        module.i = index
        module.f = from_idx
        module.type = module_cls.__name__
        layers.append(module)
        save.extend(layer % index for layer in ([from_idx] if isinstance(from_idx, int) else from_idx) if layer != -1)
        if index == 0:
            channels = []
        channels.append(layer_out_channels)

    return nn.Sequential(*layers), sorted(set(save))


class ParsedModel(nn.Module):
    """Executable graph container compatible with Ultralytics parse_model routing."""

    def __init__(self, model_cfg: dict[str, Any], num_classes: int, verbose: bool = False) -> None:
        super().__init__()
        self.yaml = dict(model_cfg)
        in_channels = int(self.yaml.get("in_channels", 1))
        self.model, self.save = parse_model(self.yaml, ch=in_channels, num_classes=num_classes, verbose=verbose)

    def forward(self, x: torch.Tensor):
        outputs: list[Any] = []
        for module in self.model:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if layer == -1 else outputs[layer] for layer in module.f]
            x = module(x)
            outputs.append(x if module.i in self.save else None)
        return x


def _resolve_module(name: str | type[nn.Module]) -> type[nn.Module]:
    if not isinstance(name, str):
        return name
    if name.startswith("nn."):
        return getattr(nn, name[3:])

    modules = {
        "AIFI": AIFI,
        "Bottleneck": Bottleneck,
        "C2f": C2f,
        "Concat": Concat,
        "Conv": Conv,
        "Detect": Detect,
        "DetectContract": DetectContract,
        "DetectDecode": DetectDecode,
        "DWConv": DWConv,
        "ResNetLayer": ResNetLayer,
        "SPPF": SPPF,
    }
    if name not in modules:
        raise KeyError(f"Unknown parsed module: {name}")
    return modules[name]