"""Detection head modules built on top of the Ultralytics-style parsed graph."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .conv import Conv, DWConv

__all__ = ["Detect", "DetectDecode", "DetectContract"]


class Detect(nn.Module):
    """Ultralytics-style dense detection head that returns raw box and score tensors."""

    def __init__(
        self,
        num_classes: int,
        ch: list[int],
        hidden_channels: int | None = None,
        legacy: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.nl = len(ch)
        if self.nl == 0:
            raise ValueError("Detect requires at least one input feature map")
        hidden_box = hidden_channels or max((16, ch[0] // 4, 64))
        hidden_cls = hidden_channels or max(ch[0], min(num_classes, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(in_channels, hidden_box, 3),
                Conv(hidden_box, hidden_box, 3),
                nn.Conv2d(hidden_box, 4, 1),
            )
            for in_channels in ch
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    Conv(in_channels, hidden_cls, 3),
                    Conv(hidden_cls, hidden_cls, 3),
                    nn.Conv2d(hidden_cls, num_classes, 1),
                )
                for in_channels in ch
            )
            if legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(in_channels, in_channels, 3), Conv(in_channels, hidden_cls, 1)),
                    nn.Sequential(DWConv(hidden_cls, hidden_cls, 3), Conv(hidden_cls, hidden_cls, 1)),
                    nn.Conv2d(hidden_cls, num_classes, 1),
                )
                for in_channels in ch
            )
        )
        self._init_predictors()

    def forward_head(
        self,
        feats: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
    ) -> dict[str, torch.Tensor]:
        box_head = self.cv2 if box_head is None else box_head
        cls_head = self.cv3 if cls_head is None else cls_head
        batch_size = feats[0].shape[0]
        boxes = torch.cat([box_head[index](feats[index]).view(batch_size, 4, -1) for index in range(self.nl)], dim=-1)
        scores = torch.cat(
            [cls_head[index](feats[index]).view(batch_size, self.num_classes, -1) for index in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": feats}

    def _init_predictors(self) -> None:
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for box_head, cls_head in zip(self.cv2, self.cv3):
            box_layer = box_head[-1]
            cls_layer = cls_head[-1]
            if box_layer.bias is not None:
                box_layer.bias.data.fill_(1.0)
            box_layer.weight.data.zero_()
            if cls_layer.bias is not None:
                cls_layer.bias.data.fill_(bias_value.item())

    def forward(self, x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        feats = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(feats) != self.nl:
            raise ValueError(f"Detect expected {self.nl} input feature maps, but received {len(feats)}")
        return self.forward_head(feats)


class DetectDecode(nn.Module):
    """Generic decoder that turns raw Detect box logits into absolute xyxy boxes."""

    def __init__(self, levels: list[dict[str, Any]]) -> None:
        super().__init__()
        if not levels:
            raise ValueError("DetectDecode requires at least one prediction level")
        self.levels_cfg = [dict(level_cfg) for level_cfg in levels]

    @staticmethod
    def _create_grid(fmp_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        hs, ws = fmp_size
        try:
            grid_y, grid_x = torch.meshgrid(torch.arange(hs, device=device), torch.arange(ws, device=device), indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(torch.arange(hs, device=device), torch.arange(ws, device=device))
        return torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)

    def _decode_boxes(self, pred_reg: torch.Tensor, fmp_size: tuple[int, int], stride: int) -> torch.Tensor:
        grid_cell = self._create_grid(fmp_size, pred_reg.device)
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * float(stride)
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * float(stride)
        pred_box = torch.cat([pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5], dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
        raw_boxes = x["boxes"]
        raw_scores = x["scores"]
        features = list(x["feats"])
        if len(features) != len(self.levels_cfg):
            raise ValueError("DetectDecode feature list and level definitions must have the same length")

        decoded_levels = []
        strides = []
        feature_shapes = []
        start = 0

        for feature, level_cfg in zip(features, self.levels_cfg):
            stride = int(level_cfg["stride"])
            fmp_size = (int(feature.shape[-2]), int(feature.shape[-1]))
            level_points = int(fmp_size[0] * fmp_size[1])
            end = start + level_points
            reg_pred = raw_boxes[:, :, start:end].transpose(1, 2).contiguous()
            decoded_levels.append(self._decode_boxes(reg_pred, fmp_size, stride))
            strides.append(stride)
            feature_shapes.append(fmp_size)
            start = end

        if start != int(raw_boxes.shape[-1]):
            raise ValueError("DetectDecode level definitions do not match the flattened prediction count")

        return {
            "boxes": torch.cat(decoded_levels, dim=1),
            "scores": raw_scores,
            "strides_all": strides,
            "fmp_sizes_all": feature_shapes,
        }


class DetectContract(nn.Module):
    """Thin adapter that converts decoded predictions into AFRIP's current loss contract."""

    def __init__(self, objectness_mode: str = "max") -> None:
        super().__init__()
        self.objectness_mode = objectness_mode

    def _aggregate_scores(self, pred_scores: torch.Tensor) -> torch.Tensor:
        if pred_scores.shape[1] == 1:
            return pred_scores.transpose(1, 2).contiguous()
        if self.objectness_mode == "max":
            return pred_scores.max(dim=1, keepdim=True)[0].transpose(1, 2).contiguous()
        if self.objectness_mode == "mean":
            return pred_scores.mean(dim=1, keepdim=True).transpose(1, 2).contiguous()
        raise ValueError(f"Unsupported objectness aggregation mode: {self.objectness_mode}")

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
        pred_box = x["boxes"]
        raw_scores = x["scores"]
        strides = list(x["strides_all"])
        feature_shapes = [tuple(fmp_size) for fmp_size in x["fmp_sizes_all"]]

        if pred_box.shape[1] != raw_scores.shape[-1]:
            raise ValueError("DetectContract decoded boxes and raw scores must have the same flattened point count")

        pred_obj = self._aggregate_scores(raw_scores)
        return {
            "pred_obj": pred_obj,
            "pred_box": pred_box,
            "strides_all": strides,
            "fmp_sizes_all": feature_shapes,
            "stride": int(strides[-1]),
            "fmp_size": tuple(feature_shapes[-1]),
        }