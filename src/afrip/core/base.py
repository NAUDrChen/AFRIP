from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """AFRIP 数据集抽象基类。"""

    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """将样本列表整理为 batch。"""


class BaseModel(nn.Module, ABC):
    """AFRIP 可训练模型基础接口。"""

    def __init__(self) -> None:
        super().__init__()
        self._training_behavior_enabled = False

    @property
    def training_behavior_enabled(self) -> bool:
        return self._training_behavior_enabled

    def set_training_behavior(self, enabled: bool) -> None:
        self._training_behavior_enabled = enabled


class BaseDetector(BaseModel, ABC):
    """AFRIP 检测器抽象基类。"""

    @abstractmethod
    def inference(self, x: torch.Tensor) -> Any:
        """执行推理并返回后处理后的结果。"""


class BaseTracker(BaseModel, ABC):
    """AFRIP 跟踪器抽象基类。"""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """根据当前观测更新跟踪状态。"""