"""YOLO 风格优化器构建工具。"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_yolo_optimizer(
    cfg: dict,
    model: nn.Module,
    resume: str | None = None,
) -> tuple[torch.optim.Optimizer, int]:
    """构建 YOLO 风格三参数组优化器，并可选地从 checkpoint 恢复。"""
    opt_type = cfg.get("optimizer", "adamw").lower()
    lr0 = cfg.get("lr0", 1e-3)
    momentum = cfg.get("momentum", 0.937)
    weight_decay = cfg.get("weight_decay", 5e-4)

    print("=" * 30)
    print(f"Optimizer: {opt_type}")
    print(f"--base lr: {lr0}")
    print(f"--momentum: {momentum}")
    print(f"--weight_decay: {weight_decay}")

    bn_types = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)

    param_groups: list[list] = [[], [], []]
    for module in model.modules():
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            param_groups[2].append(module.bias)
        if isinstance(module, bn_types):
            param_groups[1].append(module.weight)
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            param_groups[0].append(module.weight)

    if opt_type == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(param_groups[2], lr=lr0)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups[2], lr=lr0, weight_decay=0.0)
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(param_groups[2], lr=lr0, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer '{opt_type}' not implemented.")

    optimizer.add_param_group({"params": param_groups[0], "weight_decay": weight_decay})
    optimizer.add_param_group({"params": param_groups[1], "weight_decay": 0.0})

    start_epoch = 0
    if resume is not None:
        print(f"Resuming from: {resume}")
        checkpoint = torch.load(resume, map_location="cpu")
        optimizer.load_state_dict(checkpoint.pop("optimizer"))
        start_epoch = checkpoint.pop("epoch", 0)
        del checkpoint

    return optimizer, start_epoch