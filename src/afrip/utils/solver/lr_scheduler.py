"""YOLO 风格学习率调度器构建工具。"""
from __future__ import annotations

import math

import torch


def build_lr_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, object]:
    """构建 LambdaLR 学习率调度器。

    Args:
        cfg:       调度器配置字典，包含 ``scheduler`` 键（'cosine' 或 'linear'）
                   和 ``lrf`` 键（最终 lr 比例）。
        optimizer: 待调度的优化器。
        epochs:    总训练 epoch 数。

    Returns:
        (lr_scheduler, lf)  其中 lf 是 epoch → 缩放系数的函数。
    """
    sched_type = cfg.get("scheduler", "cosine").lower()
    lrf        = cfg.get("lrf", 0.01)

    print("=" * 30)
    print(f"Lr Scheduler: {sched_type}")

    if sched_type == "cosine":
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
    elif sched_type == "linear":
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf
    else:
        raise NotImplementedError(f"Scheduler '{sched_type}' not implemented.")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
