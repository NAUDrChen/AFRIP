"""Config-driven detection model assembly and high-level build helpers."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from afrip.core import BaseDetector
from afrip.modules import build_postprocessor, build_preprocessor
from .registry import (
    DETECTORS,
    build_backbone,
    build_common_block,
    build_detector,
    build_head,
    build_loss,
    build_neck,
    normalize_detector_config,
)


def _build_component(config: dict[str, Any], num_classes: int) -> nn.Module:
    component_cfg = dict(config)
    builder = component_cfg.pop("builder", "common")
    if builder == "common":
        return build_common_block(component_cfg)
    if builder == "neck":
        return build_neck(component_cfg)
    if builder == "head":
        component_cfg.setdefault("num_classes", num_classes)
        return build_head(component_cfg)
    raise ValueError(f"Unsupported component builder: {builder}")


@DETECTORS.register("ConfigurableDetectionModel")
class ConfigurableDetectionModel(BaseDetector):
    """Pure-config dense detection assembler.

    Contracts:
    - backbone: returns Tensor or sequence[Tensor]
    - feature_nodes: each node consumes named tensors or backbone outputs
    - head: returns (cls_feat, reg_feat)
    - training forward returns plain output dict
    - inference returns plain detection dict
    """

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        component_cfgs: dict[str, dict[str, Any]],
        feature_nodes: list[dict[str, Any]],
        prediction_levels: list[dict[str, Any]],
        num_classes: int = 1,
        conf_thresh: float = 0.01,
        nms_thresh: float = 0.5,
        nms_type: str = "hard",
        soft_nms_method: str = "linear",
        soft_nms_sigma: float = 0.5,
        soft_nms_score_thresh: float = 1e-3,
        topk: int | None = None,
        preprocessor_cfg: dict[str, Any] | None = None,
        postprocessor_cfg: dict[str, Any] | None = None,
        trainable: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.deploy = deploy
        self.feature_nodes_cfg = list(feature_nodes)
        self.prediction_levels_cfg = list(prediction_levels)
        self.set_training_behavior(trainable)

        if preprocessor_cfg is None:
            preprocessor_cfg = {"type": "TensorPreprocessor"}
        self.preprocessor = build_preprocessor(preprocessor_cfg)

        if postprocessor_cfg is None:
            postprocessor_cfg = {
                "type": "YOLOObjectnessPostprocessor",
                "conf_thresh": conf_thresh,
                "nms_thresh": nms_thresh,
                "nms_type": nms_type,
                "soft_nms_method": soft_nms_method,
                "soft_nms_sigma": soft_nms_sigma,
                "soft_nms_score_thresh": soft_nms_score_thresh,
                "topk": topk,
                "class_agnostic": True,
                "num_classes": 1,
            }
        self.postprocessor = build_postprocessor(postprocessor_cfg)

        self.backbone = build_backbone(backbone_cfg)
        self.components = nn.ModuleDict(
            {
                name: _build_component(cfg, num_classes=num_classes)
                for name, cfg in component_cfgs.items()
            }
        )
        self.obj_preds = nn.ModuleDict()
        self.reg_preds = nn.ModuleDict()
        for level_cfg in self.prediction_levels_cfg:
            level_name = str(level_cfg.get("name", level_cfg["feature"]))
            head_ref = level_cfg["head_ref"]
            head_module = self.components[head_ref]
            self.obj_preds[level_name] = nn.Conv2d(head_module.cls_out_dim, 1, kernel_size=1)
            self.reg_preds[level_name] = nn.Conv2d(head_module.reg_out_dim, 4, kernel_size=1)
        self._init_predictors()

    def _init_predictors(self) -> None:
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for obj_layer in self.obj_preds.values():
            if obj_layer.bias is None:
                continue
            bias = obj_layer.bias.view(1, -1)
            bias.data.fill_(bias_value.item())
            obj_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)

        for reg_layer in self.reg_preds.values():
            if reg_layer.bias is None:
                continue
            bias = reg_layer.bias.view(-1)
            bias.data.fill_(1.0)
            reg_layer.bias = nn.Parameter(bias.view(-1), requires_grad=True)
            reg_layer.weight = nn.Parameter(torch.zeros_like(reg_layer.weight), requires_grad=True)

    @staticmethod
    def _select_backbone_output(
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        index: int | None,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            if index not in (None, 0):
                raise IndexError("Backbone returned a single tensor, but indexed access was requested")
            return outputs
        if index is None:
            if len(outputs) != 1:
                raise ValueError("Backbone returned multiple tensors; source must specify an index")
            return outputs[0]
        return outputs[index]

    def _resolve_source(
        self,
        source_cfg: dict[str, Any] | str,
        backbone_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        features: dict[str, Any],
    ) -> Any:
        if isinstance(source_cfg, str):
            return features[source_cfg]

        source_type = source_cfg.get("type", "node")
        if source_type == "backbone":
            return self._select_backbone_output(backbone_outputs, source_cfg.get("index"))
        if source_type == "node":
            return features[source_cfg["ref"]]
        if source_type == "nodes":
            return [features[ref] for ref in source_cfg["refs"]]
        raise ValueError(f"Unsupported feature source type: {source_type}")

    def _apply_op(self, value: Any, op_cfg: dict[str, Any], features: dict[str, Any]) -> Any:
        op_type = op_cfg["type"]
        if op_type == "component":
            return self.components[op_cfg["ref"]](value)
        if op_type == "concat":
            if not isinstance(value, (list, tuple)):
                raise TypeError("concat op expects a list/tuple of tensors")
            return torch.cat(list(value), dim=int(op_cfg.get("dim", 1)))
        if op_type == "interpolate":
            target_ref = op_cfg.get("like")
            if target_ref is None:
                scale_factor = op_cfg.get("scale_factor")
                return F.interpolate(value, scale_factor=scale_factor, mode=op_cfg.get("mode", "nearest"))
            target = features[target_ref]
            return F.interpolate(value, size=target.shape[-2:], mode=op_cfg.get("mode", "nearest"))
        if op_type == "select":
            return value[int(op_cfg["index"])]
        raise ValueError(f"Unsupported feature op type: {op_type}")

    def _build_feature_map(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(x)
        features: dict[str, Any] = {}
        for node_cfg in self.feature_nodes_cfg:
            node_name = node_cfg["name"]
            value = self._resolve_source(node_cfg["source"], backbone_outputs, features)
            for op_cfg in node_cfg.get("ops", []):
                value = self._apply_op(value, op_cfg, features)
            features[node_name] = value
        return features

    @staticmethod
    def _create_grid(fmp_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        hs, ws = fmp_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(hs, device=device),
            torch.arange(ws, device=device),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)

    def _decode_boxes(
        self,
        pred_reg: torch.Tensor,
        fmp_size: tuple[int, int],
        stride: int,
    ) -> torch.Tensor:
        grid_cell = self._create_grid(fmp_size, pred_reg.device)
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * float(stride)
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * float(stride)
        pred_box = torch.cat([pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5], dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def _run_dense_model(self, x: torch.Tensor) -> dict[str, Any]:
        x = self.preprocessor(x)
        features = self._build_feature_map(x)
        pred_obj_levels = []
        pred_box_levels = []
        strides = []
        feature_shapes = []

        for level_cfg in self.prediction_levels_cfg:
            level_name = str(level_cfg.get("name", level_cfg["feature"]))
            feature = features[level_cfg["feature"]]
            stride = int(level_cfg["stride"])
            head = self.components[level_cfg["head_ref"]]
            cls_feat, reg_feat = head(feature)
            obj_map = self.obj_preds[level_name](cls_feat)
            reg_map = self.reg_preds[level_name](reg_feat)
            fmp_size = (int(obj_map.shape[-2]), int(obj_map.shape[-1]))

            obj_pred = obj_map.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_map.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            box_pred = self._decode_boxes(reg_pred, fmp_size, stride=stride)

            pred_obj_levels.append(obj_pred)
            pred_box_levels.append(box_pred)
            strides.append(stride)
            feature_shapes.append(fmp_size)

        pred_obj = torch.cat(pred_obj_levels, dim=1)
        pred_box = torch.cat(pred_box_levels, dim=1)
        return {
            "pred_obj": pred_obj,
            "pred_box": pred_box,
            "strides_all": strides,
            "fmp_sizes_all": feature_shapes,
            "stride": int(strides[-1]),
            "fmp_size": tuple(feature_shapes[-1]),
        }

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        outputs = self._run_dense_model(x)
        if outputs["pred_obj"].shape[0] != 1:
            raise ValueError("ConfigurableDetectionModel inference expects batch size 1")

        boxes = outputs["pred_box"][0]
        scores = outputs["pred_obj"][0].squeeze(-1).sigmoid()
        if self.deploy:
            return torch.cat([boxes, scores[:, None]], dim=-1)

        return self.postprocessor(boxes, scores)

    def forward(self, x: torch.Tensor):
        if self.training_behavior_enabled:
            return self._run_dense_model(x)
        return self.inference(x)


def assemble_detection_components(
    cfg: dict[str, Any],
    trainable: bool | None = None,
) -> tuple[Any, Any]:
    detector_cfg = normalize_detector_config(dict(cfg["detector"]))
    if trainable is not None:
        detector_cfg["trainable"] = trainable

    detector = build_detector(detector_cfg)

    loss_cfg = dict(cfg["loss"])
    if "num_classes" not in loss_cfg:
        loss_cfg["num_classes"] = detector_cfg.get("num_classes", 1)
    criterion = build_loss(loss_cfg)
    return detector, criterion