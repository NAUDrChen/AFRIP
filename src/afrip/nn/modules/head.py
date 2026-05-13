"""Detection head modules built on top of the Ultralytics-style parsed graph."""
from __future__ import annotations

import copy
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
        reg_max: int = 1,
        end2end: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.end2end = bool(end2end)
        if self.nl == 0:
            raise ValueError("Detect requires at least one input feature map")

        hidden_box = hidden_channels or max((16, ch[0] // 4, 4 * self.reg_max))
        hidden_cls = hidden_channels or max(ch[0], min(num_classes, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(in_channels, hidden_box, 3),
                Conv(hidden_box, hidden_box, 3),
                nn.Conv2d(hidden_box, 4 * self.reg_max, 1),
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
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
        self._init_predictors()

    @property
    def one2many(self) -> dict[str, nn.ModuleList]:
        return {"box_head": self.cv2, "cls_head": self.cv3}

    @property
    def one2one(self) -> dict[str, nn.ModuleList]:
        return {"box_head": self.one2one_cv2, "cls_head": self.one2one_cv3}

    def forward_head(
        self,
        feats: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
    ) -> dict[str, torch.Tensor]:
        box_head = self.cv2 if box_head is None else box_head
        cls_head = self.cv3 if cls_head is None else cls_head
        batch_size = feats[0].shape[0]
        boxes = torch.cat(
            [box_head[index](feats[index]).view(batch_size, 4 * self.reg_max, -1) for index in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[index](feats[index]).view(batch_size, self.num_classes, -1) for index in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": feats}

    def _init_predictors(self) -> None:
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        head_pairs = [(self.cv2, self.cv3)]
        if self.end2end:
            head_pairs.append((self.one2one_cv2, self.one2one_cv3))

        for box_heads, cls_heads in head_pairs:
            for box_head, cls_head in zip(box_heads, cls_heads):
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

        if self.training and self.end2end:
            one2many = self.forward_head(feats, **self.one2many)
            one2one = self.forward_head([feat.detach() for feat in feats], **self.one2one)
            return {"one2many": one2many, "one2one": one2one}

        return self.forward_head(feats, **(self.one2one if self.end2end else self.one2many))


class DetectDecode(nn.Module):
    """Decoder that turns raw Detect outputs into absolute xyxy boxes."""

    def __init__(self, levels: list[dict[str, Any]], box_mode: str = "center", reg_max: int = 1) -> None:
        super().__init__()
        if not levels:
            raise ValueError("DetectDecode requires at least one prediction level")
        if box_mode not in {"center", "dist"}:
            raise ValueError(f"Unsupported DetectDecode box_mode: {box_mode}")
        self.levels_cfg = [dict(level_cfg) for level_cfg in levels]
        self.box_mode = box_mode
        self.reg_max = int(reg_max)

    @staticmethod
    def _create_grid(fmp_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        hs, ws = fmp_size
        try:
            grid_y, grid_x = torch.meshgrid(torch.arange(hs, device=device), torch.arange(ws, device=device), indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(torch.arange(hs, device=device), torch.arange(ws, device=device))
        return torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)

    def _decode_center_boxes(self, pred_reg: torch.Tensor, fmp_size: tuple[int, int], stride: int) -> torch.Tensor:
        grid_cell = self._create_grid(fmp_size, pred_reg.device)
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * float(stride)
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * float(stride)
        pred_box = torch.cat([pred_ctr - pred_wh * 0.5, pred_ctr + pred_wh * 0.5], dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def _decode_dist_boxes(self, pred_reg: torch.Tensor, fmp_size: tuple[int, int], stride: int) -> torch.Tensor:
        grid_cell = self._create_grid(fmp_size, pred_reg.device) + 0.5
        anchor_points = grid_cell.unsqueeze(0) * float(stride)
        pred_dist = pred_reg
        if self.reg_max > 1:
            batch_size, num_points, _ = pred_reg.shape
            proj = torch.arange(self.reg_max, device=pred_reg.device, dtype=pred_reg.dtype)
            pred_dist = pred_reg.view(batch_size, num_points, 4, self.reg_max).softmax(-1).matmul(proj)
        pred_dist = pred_dist * float(stride)
        lt, rb = pred_dist.chunk(2, dim=-1)
        pred_box = torch.cat((anchor_points - lt, anchor_points + rb), dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.0
        return pred_box

    def _decode_branch(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
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
            decoder = self._decode_dist_boxes if self.box_mode == "dist" else self._decode_center_boxes
            decoded_levels.append(decoder(reg_pred, fmp_size, stride))
            strides.append(stride)
            feature_shapes.append(fmp_size)
            start = end

        if start != int(raw_boxes.shape[-1]):
            raise ValueError("DetectDecode level definitions do not match the flattened prediction count")

        return {
            "boxes": torch.cat(decoded_levels, dim=1),
            "scores": raw_scores,
            "raw_boxes": raw_boxes,
            "strides_all": strides,
            "fmp_sizes_all": feature_shapes,
        }

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "one2many" in x or "one2one" in x:
            return {key: self._decode_branch(branch) for key, branch in x.items()}
        return self._decode_branch(x)


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

    def _contract_branch(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
        pred_box = x["boxes"]
        raw_scores = x["scores"]
        raw_boxes = x["raw_boxes"]
        strides = list(x["strides_all"])
        feature_shapes = [tuple(fmp_size) for fmp_size in x["fmp_sizes_all"]]

        if pred_box.shape[1] != raw_scores.shape[-1]:
            raise ValueError("DetectContract decoded boxes and raw scores must have the same flattened point count")

        pred_obj = self._aggregate_scores(raw_scores)
        return {
            "pred_obj": pred_obj,
            "pred_box": pred_box,
            "pred_cls": raw_scores.transpose(1, 2).contiguous(),
            "pred_dist": raw_boxes.transpose(1, 2).contiguous(),
            "strides_all": strides,
            "fmp_sizes_all": feature_shapes,
            "stride": int(strides[-1]),
            "fmp_size": tuple(feature_shapes[-1]),
        }

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "one2many" in x or "one2one" in x:
            return {key: self._contract_branch(branch) for key, branch in x.items()}
        return self._contract_branch(x)