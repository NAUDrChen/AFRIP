from __future__ import annotations

from pathlib import Path

import pytest
import torch

from afrip.core import BaseDataset
from afrip.datasets import DATASETS
from afrip.engine import Trainer
from afrip.evaluation import Evaluator
from afrip.utils import load_config


SMOKE_DATASET_TYPE = "SmokeRadarDetectionDataset"


if SMOKE_DATASET_TYPE not in DATASETS.list():
    @DATASETS.register(SMOKE_DATASET_TYPE)
    class SmokeRadarDetectionDataset(BaseDataset):
        def __init__(
            self,
            image_size: tuple[int, int] = (64, 64),
            train_length: int = 2,
            test_length: int = 1,
            transforms=None,
            subset: str | None = None,
            full_frame: bool = False,
        ):
            self.image_size = tuple(image_size)
            self.train_length = int(train_length)
            self.test_length = int(test_length)
            self.transforms = transforms
            self.subset = subset
            self.full_frame = full_frame

        def __len__(self) -> int:
            if self.subset == "test":
                return self.test_length
            return self.train_length

        def __getitem__(self, index: int):
            height, width = self.image_size
            image = torch.zeros((1, height, width), dtype=torch.float32)
            offset = float((index % 2) * 4)
            x1 = 8.0 + offset
            y1 = 8.0
            x2 = 24.0 + offset
            y2 = 24.0
            image[:, int(y1):int(y2), int(x1):int(x2)] = 1.0

            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.tensor([index % 3], dtype=torch.long)

            if self.transforms is not None:
                image, boxes = self.transforms(image, boxes)

            return {
                "image": image,
                "boxes": boxes,
                "labels": labels,
                "meta": {
                    "file": f"{self.subset or 'train'}_{index}.mat",
                    "global_origin": (0, 0),
                    "index": index,
                },
            }

        @staticmethod
        def collate_fn(batch):
            return {
                "images": torch.stack([sample["image"] for sample in batch], dim=0),
                "targets": [
                    {
                        "boxes": sample["boxes"],
                        "labels": sample["labels"],
                    }
                    for sample in batch
                ],
                "batch_meta": [sample["meta"] for sample in batch],
            }


@pytest.mark.parametrize(
    "config_name",
    [
        "radardet_rdcnn_sort.yaml",
        "radardet_yolortv2_sort.yaml",
    ],
)
def test_detection_training_and_validation_smoke(tmp_path: Path, config_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / config_name
    cfg = load_config(config_path)

    cfg["runtime"]["device"] = "cpu"
    cfg["runtime"]["log_interval"] = 1000
    cfg["dataloader"]["batch_size"] = 1
    cfg["dataloader"]["shuffle"] = False
    cfg["dataset"] = {
        "type": SMOKE_DATASET_TYPE,
        "image_size": (64, 64),
        "train_length": 2,
        "test_length": 1,
    }
    cfg["train_transforms"] = []
    cfg["val_transforms"] = []
    cfg["strategy"]["train"]["max_epoch"] = 1
    cfg["strategy"]["train"]["eval_interval"] = 1
    cfg["strategy"]["train"]["num_workers_train"] = 0
    cfg["strategy"]["train"]["num_workers_test"] = 0
    cfg["strategy"]["train"]["persistent_workers"] = False
    cfg["strategy"]["train"]["fp16"] = False
    cfg["strategy"]["eval"]["save_folder"] = str(tmp_path / "weights")

    detector_cfg = cfg["detector"]
    detector_cfg["postprocessor_cfg"]["conf_thresh"] = 0.0
    detector_cfg["postprocessor_cfg"]["nms_thresh"] = 0.5

    trainer = Trainer(cfg)
    evaluator = Evaluator(cfg)

    train_batch = next(iter(trainer.train_loader))
    assert "images" in train_batch
    assert "targets" in train_batch
    assert "raw_boxes" not in train_batch
    assert train_batch["images"].dtype == torch.float32
    assert train_batch["targets"][0]["boxes"].shape[-1] == 4

    best_map = trainer.run(evaluator=evaluator)

    assert isinstance(best_map, float)
    assert best_map >= 0.0
    saved_weights = list((tmp_path / "weights").glob("*.pth"))
    assert saved_weights