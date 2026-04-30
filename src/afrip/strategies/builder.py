from __future__ import annotations

import torch
import torch.nn as nn

from .lr_scheduler import build_lr_scheduler
from .optimizer import build_yolo_optimizer


def build_optimizer(
    cfg: dict,
    model: nn.Module,
    resume: str | None = None,
) -> tuple[torch.optim.Optimizer, int]:
    """策略层正式优化器构建入口。"""
    return build_yolo_optimizer(cfg, model, resume)


def build_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    epochs: int,
):
    """策略层正式学习率调度器构建入口。"""
    return build_lr_scheduler(cfg, optimizer, epochs)