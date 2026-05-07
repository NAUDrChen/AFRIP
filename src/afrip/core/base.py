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
    """AFRIP 可训练模型基础接口。

    训练/推理分支统一复用 nn.Module 的 train()/eval() 状态。
    """


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