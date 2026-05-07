"""Registered dense detection heads for AFRIP models."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from afrip.models.registry import HEADS

from .decoupled_head import DecoupledHead


@HEADS.register("DenseDetectionHead")
class DenseDetectionHead(nn.Module):
    """Shared decoupled dense head that predicts objectness and boxes on named levels."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        levels: list[dict[str, Any]],
        num_classes: int,
        num_cls_head: int = 2,
        num_reg_head: int = 2,
        act_type: str = "lrelu",
        norm_type: str = "BN",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        if not levels:
            raise ValueError("DenseDetectionHead requires at least one prediction level")
        self.levels_cfg = [dict(level_cfg) for level_cfg in levels]
        self.feature_head = DecoupledHead(
            in_dim=in_dim,
            out_dim=out_dim,
            num_classes=num_classes,
            num_cls_head=num_cls_head,
            num_reg_head=num_reg_head,
            act_type=act_type,
            norm_type=norm_type,
            depthwise=depthwise,
        )
        self.obj_preds = nn.ModuleDict()
        self.reg_preds = nn.ModuleDict()
        for level_cfg in self.levels_cfg:
            level_name = str(level_cfg.get("name") or level_cfg["feature"])
            self.obj_preds[level_name] = nn.Conv2d(
                self.feature_head.cls_out_dim,
                1,
                kernel_size=1,
            )
            self.reg_preds[level_name] = nn.Conv2d(
                self.feature_head.reg_out_dim,
                4,
                kernel_size=1,
            )
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
            reg_layer.weight = nn.Parameter(
                torch.zeros_like(reg_layer.weight),
                requires_grad=True,
            )

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
        pred_box = torch.cat(
            [pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5],
            dim=-1,
        )
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, Any]:
        pred_obj_levels = []
        pred_box_levels = []
        strides = []
        feature_shapes = []

        for level_cfg in self.levels_cfg:
            level_name = str(level_cfg.get("name") or level_cfg["feature"])
            feature_name = str(level_cfg.get("feature", level_name))
            if feature_name not in features:
                raise KeyError(f"Missing feature '{feature_name}' for level '{level_name}'")
            feature = features[feature_name]
            stride = int(level_cfg["stride"])
            cls_feat, reg_feat = self.feature_head(feature)
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