from __future__ import annotations

import pytest
import torch

from afrip.core import BaseDataset, BaseDetector
from afrip.engine import DetectionRunner
from afrip.models import build_detector
from afrip.strategies import build_optimizer, build_scheduler


class DummyDataset(BaseDataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return {"value": index}

    @staticmethod
    def collate_fn(batch):
        return {"items": batch}


def test_base_dataset_collate_structure() -> None:
    batch = DummyDataset.collate_fn([{"value": 1}, {"value": 2}])
    assert batch == {"items": [{"value": 1}, {"value": 2}]}


def test_radar_window_dataset_is_base_dataset() -> None:
    from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset

    assert issubclass(RadarWindowDataset, BaseDataset)


def test_configurable_detection_model_is_base_detector() -> None:
    from afrip.models.common import ConfigurableDetectionModel

    assert issubclass(ConfigurableDetectionModel, BaseDetector)


def test_build_detector_from_explicit_config() -> None:
    config = {
        "type": "ConfigurableDetectionModel",
        "num_classes": 3,
        "preprocessor_cfg": {"type": "TensorPreprocessor"},
        "postprocessor_cfg": {
            "type": "YOLOObjectnessPostprocessor",
            "conf_thresh": 0.01,
            "nms_thresh": 0.5,
            "class_agnostic": True,
            "num_classes": 1,
        },
        "backbone_cfg": {"type": "ResNet18", "in_channels": 1},
        "neck_cfg": {"type": "SingleScaleSPPFNeck", "in_dim": 256, "out_dim": 512},
        "head_cfg": {
            "type": "DenseDetectionHead",
            "in_dim": 512,
            "out_dim": 512,
            "levels": [{"name": "p3", "stride": 16}],
        },
    }
    detector = build_detector(config)

    assert isinstance(detector, BaseDetector)


def test_build_detector_requires_explicit_preprocessor_and_postprocessor() -> None:
    config = {
        "type": "ConfigurableDetectionModel",
        "backbone_cfg": {"type": "ResNet18", "in_channels": 1},
        "neck_cfg": {"type": "SingleScaleSPPFNeck", "in_dim": 256, "out_dim": 512},
        "head_cfg": {
            "type": "DenseDetectionHead",
            "in_dim": 512,
            "out_dim": 512,
            "levels": [{"name": "p3", "stride": 16}],
        },
    }

    with pytest.raises(TypeError):
        build_detector(config)


def test_strategies_exports_builders() -> None:
    assert callable(build_optimizer)
    assert callable(build_scheduler)


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