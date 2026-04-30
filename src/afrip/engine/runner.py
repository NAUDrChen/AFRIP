"""统一运行入口骨架。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from afrip.evaluation import Evaluator

from .trainer import Trainer


class BaseRunner(ABC):
    """AFRIP 运行器抽象基类。"""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def run(self) -> Any:
        """执行一次完整运行。"""


class DetectionRunner(BaseRunner):
    """面向检测任务的默认运行器。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        trainer_factory: Callable[[dict[str, Any]], Trainer] = Trainer,
        evaluator_factory: Callable[[dict[str, Any]], Evaluator] | None = Evaluator,
    ) -> None:
        super().__init__(cfg)
        self.trainer = trainer_factory(cfg)
        self.evaluator = evaluator_factory(cfg) if evaluator_factory is not None else None

    def run(self) -> float:
        return self.trainer.run(evaluator=self.evaluator)