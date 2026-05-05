from __future__ import annotations

import torch

from afrip.core import BaseDataset, BaseDetector
from afrip.engine import DetectionRunner
from afrip.models import normalize_detector_config
from afrip.strategies import build_optimizer, build_scheduler


class DummyDataset(BaseDataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return {"value": index}

    @staticmethod
    def collate_fn(batch):
        return {"items": batch}


def test_base_dataset_collate_contract() -> None:
    batch = DummyDataset.collate_fn([{"value": 1}, {"value": 2}])
    assert batch == {"items": [{"value": 1}, {"value": 2}]}


def test_radar_window_dataset_is_base_dataset() -> None:
    from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset

    assert issubclass(RadarWindowDataset, BaseDataset)


def test_unified_dense_detector_is_base_detector() -> None:
    from afrip.models.detectors.assembly import UnifiedDenseDetector

    assert issubclass(UnifiedDenseDetector, BaseDetector)


def test_legacy_detector_types_normalize_to_unified_assembly() -> None:
    normalized = normalize_detector_config({"type": "YOLORTv2", "num_classes": 3})

    assert normalized["type"] == "UnifiedDenseDetector"
    assert normalized["architecture"] == "yolort_v2"


def test_strategies_exports_builders() -> None:
    assert callable(build_optimizer)
    assert callable(build_scheduler)


def test_legacy_solver_wrappers_still_work() -> None:
    from afrip.utils.solver import build_lr_scheduler as legacy_scheduler
    from afrip.utils.solver import build_yolo_optimizer as legacy_optimizer

    assert callable(legacy_optimizer)
    assert callable(legacy_scheduler)


def test_detection_runner_uses_trainer_and_evaluator_factories() -> None:
    class DummyTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, evaluator=None):
            assert evaluator == "evaluator"
            return 0.75

    cfg = {"experiment": {"name": "demo"}}
    runner = DetectionRunner(
        cfg,
        trainer_factory=DummyTrainer,
        evaluator_factory=lambda current_cfg: "evaluator",
    )

    assert runner.run() == 0.75


def test_build_optimizer_and_scheduler_from_new_strategy_modules() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    optimizer, start_epoch = build_optimizer(
        {"optimizer": "adamw", "lr0": 1e-3, "momentum": 0.9, "weight_decay": 1e-4},
        model,
    )
    scheduler, schedule_fn = build_scheduler(
        {"scheduler": "linear", "lrf": 0.1},
        optimizer,
        epochs=10,
    )

    assert start_epoch == 0
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert hasattr(scheduler, "step")
    assert callable(schedule_fn)