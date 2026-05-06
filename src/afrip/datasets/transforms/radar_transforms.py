from __future__ import annotations

import math
from typing import Tuple
import torch

from afrip.datasets.registry import Compose, TRANSFORMS

@TRANSFORMS.register("AmplitudeNormalize")
class AmplitudeNormalize:
    """幅度归一化。
    mode:
      db_percentile: 计算 |A| -> dB -> 按 p1,p99 分位线拉伸到 [0,1]
      minmax: |A| 按全局 min,max 缩放到 [0,1]
      standard: |A| 标准化 (减均值除标准差) 后再裁剪到 [-k,k] 映射到 [0,1]
    对 complex_mode='stack' 时保持相位（通过幅度重标定 real/imag）。
    """
    def __init__(self, mode: str = "db_percentile", clip_val: float = 3.0, eps: float = 1e-12):
        self.mode = mode
        self.clip_val = clip_val
        self.eps = eps

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor):
        # image: [C,H,W]
        image = image.float()
        if image.shape[0] == 2:
            real = image[0]
            imag = image[1]
            amp = torch.sqrt(real.square() + imag.square())
            phase = torch.atan2(imag, real)
            is_stack = True
        else:
            amp = image[0].abs()
            phase = None
            is_stack = False

        if self.mode == "db_percentile":
            db = 20.0 * torch.log10(amp + self.eps)
            p1 = torch.quantile(db.reshape(-1), 0.01)
            p99 = torch.quantile(db.reshape(-1), 0.99)
            norm = ((db - p1) / (p99 - p1 + 1e-6)).clamp(0.0, 1.0)
        elif self.mode == "minmax":
            a_min = amp.min()
            a_max = amp.max()
            norm = (amp - a_min) / (a_max - a_min + 1e-6)
        elif self.mode == "standard":
            mean = amp.mean()
            std = amp.std() + 1e-6
            z = (amp - mean) / std
            z = z.clamp(-self.clip_val, self.clip_val)
            norm = (z + self.clip_val) / (2 * self.clip_val)
        else:
            raise ValueError(f"未知归一化模式: {self.mode}")

        if is_stack:
            real_new = torch.cos(phase) * norm
            imag_new = torch.sin(phase) * norm
            image = torch.stack([real_new, imag_new], dim=0)
        else:
            image = norm.unsqueeze(0)

        return image.to(dtype=torch.float32), boxes
    
@TRANSFORMS.register("RandomVerticalFlip")
class RandomVerticalFlip:
    """按概率翻转 Y 维(方位)；更新 raw_boxes 的 y1,y2。"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor):
        if torch.rand(1).item() >= self.prob:
            return image, boxes
        _, height, _ = image.shape
        image = torch.flip(image, dims=[1])
        if boxes.numel() > 0:
            boxes = boxes.clone()
            y1 = boxes[:, 1].clone()
            y2 = boxes[:, 3].clone()
            boxes[:, 1] = height - y2
            boxes[:, 3] = height - y1
        return image, boxes
    
@TRANSFORMS.register("SeaClutterInjection")
class SeaClutterInjection:
    """海杂波注入。
    方法:
      - 随机目标 SNR (dB) 于区间 snr_db_range
      - 估计当前信号功率 (默认: 目标框区域内幅度均值平方作为功率；若无框用全局)
      - 计算噪声功率 = signal_power / 10^(SNR/10)
      - 生成噪声:
          complex: real/imag ~ N(0, sigma^2/2) 使得复功率 ~ sigma^2
          magnitude-only: Rayleigh 近似: R = sqrt(X^2+Y^2), X,Y~N(0, sigma^2)
      - 可选择是否仅对背景区域添加噪声 (background_only)
    """
    def __init__(
        self,
        prob: float = 0.7,
        snr_db_range: Tuple[float, float] = (5.0, 20.0),
        background_only: bool = False,
        eps: float = 1e-12
    ):
        self.prob = prob
        self.snr_db_range = snr_db_range
        self.background_only = background_only
        self.eps = eps

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor):
        if torch.rand(1).item() >= self.prob:
            return image, boxes

        image = image.float()
        channels, height, width = image.shape

        if channels == 2:
            real = image[0].clone()
            imag = image[1].clone()
            amp = torch.sqrt(real.square() + imag.square())
            is_complex = True
        else:
            amp = image[0].abs()
            is_complex = False

        mask = None
        if boxes.numel() > 0:
            mask = torch.zeros((height, width), dtype=torch.bool, device=image.device)
            for box in boxes:
                x1 = int(box[0].clamp(0, width - 1).item())
                y1 = int(box[1].clamp(0, height - 1).item())
                x2 = int(box[2].clamp(0, width).item())
                y2 = int(box[3].clamp(0, height).item())
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True

        if mask is not None and mask.any():
            signal_power = float(amp[mask].square().mean().item() + self.eps)
        else:
            signal_power = float(amp.square().mean().item() + self.eps)

        snr_db = torch.empty(1).uniform_(*self.snr_db_range).item()
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_lin + self.eps)
        sigma = math.sqrt(noise_power)

        if is_complex:
            sigma_half = sigma / math.sqrt(2.0)
            noise_real = torch.randn((height, width), dtype=image.dtype, device=image.device) * sigma_half
            noise_imag = torch.randn((height, width), dtype=image.dtype, device=image.device) * sigma_half
            if self.background_only and mask is not None:
                bg_mask = ~mask
                real[bg_mask] += noise_real[bg_mask]
                imag[bg_mask] += noise_imag[bg_mask]
            else:
                real += noise_real
                imag += noise_imag
            image = torch.stack([real, imag], dim=0)
        else:
            sigma_half = sigma / math.sqrt(2.0)
            noise_real = torch.randn((height, width), dtype=image.dtype, device=image.device) * sigma_half
            noise_imag = torch.randn((height, width), dtype=image.dtype, device=image.device) * sigma_half
            noise_amp = torch.sqrt(noise_real.square() + noise_imag.square())
            image = image.clone()
            if self.background_only and mask is not None:
                bg_mask = ~mask
                image[0][bg_mask] += noise_amp[bg_mask]
            else:
                image[0] += noise_amp

        return image.to(dtype=torch.float32), boxes
    
@TRANSFORMS.register("PadToStride")
class PadToStride:
    """将输入 [C,H,W] 通过右侧与下侧填充对齐到 stride 的最小倍数，不缩放图像。推荐在归一化后进行填充，避免归一化对填充值的再处理。
    - 仅填充：不改变原像素位置，左/上对齐
    - 更新 raw_boxes: x/y 坐标保持不变；x2/y2 不变；无需平移
    - 填充值:
        complex(stack)通道: 用 0 填充 real/imag
        幅度/归一化通道: 默认 0.0
    """
    def __init__(self, stride: int = 32, pad_value: float = 0.0):
        assert stride > 0
        self.stride = stride
        self.pad_value = pad_value

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor):
        # image: [C,H,W]
        C, H, W = image.shape
        s = self.stride

        def _ceil_to_stride(x: int, s: int) -> int:
            return ((x + s - 1) // s) * s

        H_pad = _ceil_to_stride(H, s)
        W_pad = _ceil_to_stride(W, s)

        if H_pad == H and W_pad == W:
            # 已经是 stride 倍数，直接返回
            return image, boxes

        pad_bottom = H_pad - H
        pad_right = W_pad - W

        # 构造填充张量：仅在下/右进行填充，左/上不填充
        if C == 2:
            # complex stack: real/imag 填 0
            pad_tensor = torch.zeros((C, H_pad, W_pad), dtype=image.dtype)
        else:
            pad_tensor = torch.full((C, H_pad, W_pad), fill_value=self.pad_value, dtype=image.dtype)

        # 将原图拷贝到左上角
        pad_tensor[:, :H, :W] = image
        image = pad_tensor

        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, 0] = boxes[:, 0].clamp(0, W_pad - 1)
            boxes[:, 2] = boxes[:, 2].clamp(0, W_pad)
            boxes[:, 1] = boxes[:, 1].clamp(0, H_pad - 1)
            boxes[:, 3] = boxes[:, 3].clamp(0, H_pad)

        return image, boxes


# 便捷函数：构造典型增强流水线
def build_radar_augmentation(
    norm_mode: str = "db_percentile",
    vertical_flip_prob: float = 0.5,
    clutter_prob: float = 0.7,
    snr_range: Tuple[float, float] = (5.0, 20.0),
    background_only: bool = False
):
    return Compose([
        SeaClutterInjection(prob=clutter_prob,
                            snr_db_range=snr_range,
                            background_only=background_only),
        RandomVerticalFlip(prob=vertical_flip_prob),
        AmplitudeNormalize(mode=norm_mode)
    ])