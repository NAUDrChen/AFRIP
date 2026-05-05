"""Dense detector shared outputs and execution flow."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from afrip.core import BaseDetector
from afrip.modules import build_postprocessor, build_preprocessor


class DenseDetectionOutputs:
    """Container for normalized dense detector outputs."""

    def __init__(
        self,
        pred_obj: torch.Tensor,
        pred_box: torch.Tensor,
        stride: int,
        fmp_size: tuple[int, int],
        strides_all: list[int] | None = None,
        fmp_sizes_all: list[tuple[int, int]] | None = None,
    ) -> None:
        self.pred_obj = pred_obj
        self.pred_box = pred_box
        self.stride = int(stride)
        self.fmp_size = tuple(fmp_size)
        self.strides_all = list(strides_all or [int(stride)])
        self.fmp_sizes_all = list(fmp_sizes_all or [tuple(fmp_size)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "pred_obj": self.pred_obj,
            "pred_box": self.pred_box,
            "stride": self.stride,
            "fmp_size": self.fmp_size,
            "strides_all": self.strides_all,
            "fmp_sizes_all": self.fmp_sizes_all,
        }

    def to_legacy_training_output(self) -> dict[str, Any]:
        return self.as_dict()


def normalize_dense_outputs(
    outputs: DenseDetectionOutputs | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(outputs, DenseDetectionOutputs):
        return outputs.as_dict()
    return outputs


class DenseDetectorBase(BaseDetector, ABC):
    """Shared preprocess, decode, training and inference flow."""

    def __init__(
        self,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        preprocessor_cfg: dict[str, Any] | None = None,
        postprocessor_cfg: dict[str, Any] | None = None,
        trainable: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.deploy = deploy
        self.set_training_behavior(trainable)

        if preprocessor_cfg is None:
            preprocessor_cfg = {"type": "TensorPreprocessor"}
        self.preprocessor = build_preprocessor(preprocessor_cfg)

        if postprocessor_cfg is None:
            postprocessor_cfg = {
                "type": "YOLOObjectnessPostprocessor",
                "conf_thresh": conf_thresh,
                "nms_thresh": nms_thresh,
                "class_agnostic": True,
                "num_classes": 1,
            }
        self.postprocessor = build_postprocessor(postprocessor_cfg)

    def create_grid(
        self,
        fmp_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        hs, ws = fmp_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(hs, device=device),
            torch.arange(ws, device=device),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)

    def decode_boxes(
        self,
        pred_reg: torch.Tensor,
        fmp_size: tuple[int, int],
        stride: int,
    ) -> torch.Tensor:
        grid_cell = self.create_grid(fmp_size, pred_reg.device)
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * float(stride)
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * float(stride)
        pred_box = torch.cat(
            [pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5],
            dim=-1,
        )
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def _flatten_predictions(
        self,
        obj_map: torch.Tensor,
        reg_map: torch.Tensor,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        fmp_size = tuple(obj_map.shape[-2:])
        obj_pred = obj_map.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_map.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        box_pred = self.decode_boxes(reg_pred, fmp_size, stride=stride)
        return obj_pred, box_pred, fmp_size

    @abstractmethod
    def _forward_levels(
        self,
        x: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        """Return [(obj_map, reg_map, stride), ...]."""

    def _collect_outputs(self, x: torch.Tensor) -> DenseDetectionOutputs:
        x = self.preprocessor(x)
        levels = self._forward_levels(x)

        pred_obj_levels = []
        pred_box_levels = []
        strides_all = []
        fmp_sizes_all = []
        for obj_map, reg_map, stride in levels:
            pred_obj, pred_box, fmp_size = self._flatten_predictions(
                obj_map,
                reg_map,
                stride,
            )
            pred_obj_levels.append(pred_obj)
            pred_box_levels.append(pred_box)
            strides_all.append(int(stride))
            fmp_sizes_all.append(fmp_size)

        return DenseDetectionOutputs(
            pred_obj=torch.cat(pred_obj_levels, dim=1),
            pred_box=torch.cat(pred_box_levels, dim=1),
            stride=strides_all[-1],
            fmp_size=fmp_sizes_all[-1],
            strides_all=strides_all,
            fmp_sizes_all=fmp_sizes_all,
        )

    @torch.no_grad()
    def inference(self, x: torch.Tensor):
        outputs = self._collect_outputs(x)
        if outputs.pred_obj.shape[0] != 1:
            raise ValueError("Dense detector inference expects batch size 1")

        bboxes = outputs.pred_box[0]
        obj_scores = outputs.pred_obj[0].squeeze(-1).sigmoid()
        if self.deploy:
            return torch.cat([bboxes, obj_scores[:, None]], dim=-1)

        return self.postprocessor(
            bboxes.cpu().numpy(),
            obj_scores.cpu().numpy(),
        )

    def forward(self, x: torch.Tensor):
        outputs = self._collect_outputs(x)
        if self.training_behavior_enabled:
            return outputs.to_legacy_training_output()
        return self.inference(x)