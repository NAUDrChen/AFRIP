"""兼容入口：请改用 `afrip.strategies.lr_scheduler`。"""
from afrip.strategies.lr_scheduler import build_lr_scheduler

__all__ = ["build_lr_scheduler"]
