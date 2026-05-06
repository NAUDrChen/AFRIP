"""解耦检测头（分类 + 回归并行特征分支）。"""
from __future__ import annotations

import torch
import torch.nn as nn

from afrip.models.common import HEADS, Conv


@HEADS.register("DecoupledHead")
class DecoupledHead(nn.Module):
    """解耦检测头：分类分支与回归分支独立。

    Args:
        in_dim: 输入通道数。
        out_dim: 基础输出通道数（分类 ≥ num_classes，回归 ≥ 64）。
        num_classes: 目标类别数。
        num_cls_head: 分类分支卷积层数，默认 2。
        num_reg_head: 回归分支卷积层数，默认 2。
        act_type: 激活函数类型，默认 'lrelu'。
        norm_type: 归一化类型，默认 'BN'。
        depthwise: 是否使用 depthwise 卷积，默认 False。
    """

    def __init__(self, in_dim: int, out_dim: int, num_classes: int,
                 num_cls_head: int = 2, num_reg_head: int = 2,
                 act_type: str = 'lrelu', norm_type: str = 'BN',
                 depthwise: bool = False):
        super().__init__()
        self.in_dim      = in_dim
        self.act_type    = act_type
        self.norm_type   = norm_type

        # 分类分支
        self.cls_out_dim = max(out_dim, num_classes)
        cls_feats: list[nn.Module] = []
        for i in range(num_cls_head):
            c_in = in_dim if i == 0 else self.cls_out_dim
            cls_feats.append(
                Conv(c_in, self.cls_out_dim, k=3, p=1, s=1,
                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.cls_feats = nn.Sequential(*cls_feats)

        # 回归分支
        self.reg_out_dim = max(out_dim, 64)
        reg_feats: list[nn.Module] = []
        for i in range(num_reg_head):
            c_in = in_dim if i == 0 else self.reg_out_dim
            reg_feats.append(
                Conv(c_in, self.reg_out_dim, k=3, p=1, s=1,
                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.reg_feats = nn.Sequential(*reg_feats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (cls_feats, reg_feats)，形状均为 [B, C, H, W]。"""
        return self.cls_feats(x), self.reg_feats(x)
