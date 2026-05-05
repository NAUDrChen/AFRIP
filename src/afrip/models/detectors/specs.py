"""Architecture-specific dense detection assembly blocks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from afrip.models.common import Conv
from afrip.models.registry import build_backbone, build_head, build_neck


class DenseAssemblySpec(nn.Module, ABC):
    """Owns architecture-specific forward path and per-level heads."""

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        neck_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        num_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.neck = build_neck(neck_cfg)
        self.head_dim = self.neck.out_dim
        merged_head_cfg = {
            **head_cfg,
            "in_dim": self.head_dim,
            "out_dim": self.head_dim,
            "num_classes": num_classes,
        }
        self.heads = nn.ModuleList(
            [build_head(merged_head_cfg) for _ in range(self.num_levels)]
        )
        self.strides = list(self.default_strides)

    @property
    @abstractmethod
    def default_strides(self) -> list[int]:
        raise NotImplementedError

    @property
    def num_levels(self) -> int:
        return len(self.default_strides)

    @abstractmethod
    def build_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        features = self.build_features(x)
        outputs = []
        for feature, head, stride in zip(features, self.heads, self.strides):
            cls_feat, reg_feat = head(feature)
            outputs.append((cls_feat, reg_feat, int(stride)))
        return outputs


class SingleScaleDenseSpec(DenseAssemblySpec):
    @property
    def default_strides(self) -> list[int]:
        return [16]

    def build_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.neck(self.backbone(x))]


class PyramidP2P3DenseSpec(DenseAssemblySpec):
    @property
    def default_strides(self) -> list[int]:
        return [8, 16]

    def __init__(
        self,
        backbone_cfg: dict[str, Any],
        neck_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        num_classes: int,
    ) -> None:
        super().__init__(backbone_cfg, neck_cfg, head_cfg, num_classes)
        backbone_channels = getattr(self.backbone, "out_channels", None)
        if not isinstance(backbone_channels, (tuple, list)) or len(backbone_channels) < 1:
            raise ValueError("PyramidP2P3DenseSpec requires pyramid backbone out_channels")

        act_type = neck_cfg.get("act_type", "lrelu")
        norm_type = neck_cfg.get("norm_type", "BN")
        self.p2_lateral = Conv(
            int(backbone_channels[0]),
            self.head_dim,
            k=1,
            p=0,
            s=1,
            act_type=act_type,
            norm_type=norm_type,
        )
        self.p2_fuse = Conv(
            self.head_dim * 2,
            self.head_dim,
            k=3,
            p=1,
            s=1,
            act_type=act_type,
            norm_type=norm_type,
        )

    def build_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        c3, c4 = self.backbone(x)
        p3 = self.neck(c4)
        p2_lat = self.p2_lateral(c3)
        p3_up = F.interpolate(p3, size=p2_lat.shape[-2:], mode="nearest")
        p2 = self.p2_fuse(torch.cat([p2_lat, p3_up], dim=1))
        return [p2, p3]


def build_dense_spec(
    architecture: str,
    backbone_cfg: dict[str, Any],
    neck_cfg: dict[str, Any],
    head_cfg: dict[str, Any],
    num_classes: int,
) -> DenseAssemblySpec:
    normalized = architecture.lower()
    spec_cls = {
        "single_scale": SingleScaleDenseSpec,
        "yolort_v1": SingleScaleDenseSpec,
        "p2p3": PyramidP2P3DenseSpec,
        "yolort_v2": PyramidP2P3DenseSpec,
    }.get(normalized)
    if spec_cls is None:
        raise ValueError(f"Unknown dense detector architecture: {architecture}")
    return spec_cls(backbone_cfg, neck_cfg, head_cfg, num_classes)