from __future__ import annotations

from typing import Tuple
import numpy as np
import torch

from afrip.datasets.registry import TRANSFORMS

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

    def __call__(self, image: torch.Tensor, raw_boxes: np.ndarray):
        # image: [C,H,W]
        C, H, W = image.shape
        img_np = image.cpu().numpy()
        if C == 2:  # stack 或 magnitude_phase
            # 假设通道0=real或幅度；尝试判断是否复栈: 用方差比简单区分
            # 对 stack: 复数 => real/imag
            real = img_np[0]
            imag = img_np[1]
            amp = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)
            is_stack = True
        else:
            amp = np.abs(img_np[0])
            phase = None
            is_stack = False

        if self.mode == "db_percentile":
            db = 20.0 * np.log10(amp + self.eps)
            p1, p99 = np.percentile(db, [1, 99])
            norm = np.clip((db - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
        elif self.mode == "minmax":
            a_min, a_max = float(amp.min()), float(amp.max())
            norm = (amp - a_min) / (a_max - a_min + 1e-6)
        elif self.mode == "standard":
            mean = float(amp.mean())
            std = float(amp.std() + 1e-6)
            z = (amp - mean) / std
            z = np.clip(z, -self.clip_val, self.clip_val)
            norm = (z + self.clip_val) / (2 * self.clip_val)
        else:
            raise ValueError(f"未知归一化模式: {self.mode}")

        if is_stack:
            # 保持相位: real_norm = cos(phase)*norm, imag_norm = sin(phase)*norm
            real_new = np.cos(phase) * norm
            imag_new = np.sin(phase) * norm
            out = np.stack([real_new, imag_new], axis=0).astype(np.float32)
        else:
            out = norm[None, ...].astype(np.float32)

        image = torch.from_numpy(out)
        return image, raw_boxes
    
@TRANSFORMS.register("RandomVerticalFlip")
class RandomVerticalFlip:
    """按概率翻转 Y 维(方位)；更新 raw_boxes 的 y1,y2。"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, raw_boxes: np.ndarray):
        if np.random.rand() >= self.prob:
            return image, raw_boxes
        C, H, W = image.shape
        # 翻转
        image = torch.flip(image, dims=[1])  # Y 维
        if raw_boxes.shape[0] > 0:
            # y 相关: new_y1 = H - old_y2, new_y2 = H - old_y1
            y1 = raw_boxes[:, 3].copy()
            y2 = raw_boxes[:, 5].copy()
            raw_boxes[:, 3] = H - y2
            raw_boxes[:, 5] = H - y1
        return image, raw_boxes
    
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

    def __call__(self, image: torch.Tensor, raw_boxes: np.ndarray):
        if np.random.rand() >= self.prob:
            return image, raw_boxes

        C, H, W = image.shape
        img_np = image.cpu().numpy()

        if C == 2:  # 复栈 real/imag
            real = img_np[0]
            imag = img_np[1]
            amp = np.sqrt(real**2 + imag**2)
            is_complex = True
        else:
            amp = np.abs(img_np[0])
            is_complex = False

        # 计算目标区域掩码
        if raw_boxes.shape[0] > 0:
            mask = np.zeros((H, W), dtype=bool)
            for b in raw_boxes:
                x1, y1, x2, y2 = int(b[2]), int(b[3]), int(b[4]), int(b[5])
                x1 = max(0, min(W - 1, x1))
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1))
                y2 = max(0, min(H, y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
            if mask.any():
                signal_power = float((amp[mask]**2).mean() + self.eps)
            else:
                signal_power = float((amp**2).mean() + self.eps)
        else:
            signal_power = float((amp**2).mean() + self.eps)

        snr_db = np.random.uniform(*self.snr_db_range)
        snr_lin = 10.0**(snr_db / 10.0)
        noise_power = signal_power / (snr_lin + self.eps)
        # 复噪声 sigma^2 = noise_power => real/imag 方差 = noise_power/2
        sigma = np.sqrt(noise_power)

        if is_complex:
            noise_real = np.random.normal(0.0, sigma / np.sqrt(2), size=(H, W))
            noise_imag = np.random.normal(0.0, sigma / np.sqrt(2), size=(H, W))
            if self.background_only and raw_boxes.shape[0] > 0:
                bg_mask = ~mask
                real[bg_mask] += noise_real[bg_mask]
                imag[bg_mask] += noise_imag[bg_mask]
            else:
                real += noise_real
                imag += noise_imag
            out = np.stack([real, imag], axis=0).astype(np.float32)
        else:
            # 幅度噪声: 生成复噪声取幅度 -> Rayleigh 近似
            noise_real = np.random.normal(0.0, sigma / np.sqrt(2), size=(H, W))
            noise_imag = np.random.normal(0.0, sigma / np.sqrt(2), size=(H, W))
            noise_amp = np.sqrt(noise_real**2 + noise_imag**2)
            if self.background_only and raw_boxes.shape[0] > 0:
                bg_mask = ~mask
                img_np[0][bg_mask] += noise_amp[bg_mask]
            else:
                img_np[0] += noise_amp
            out = img_np.astype(np.float32)

        image = torch.from_numpy(out)
        return image, raw_boxes
    
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

    def __call__(self, image: torch.Tensor, raw_boxes: np.ndarray):
        # image: [C,H,W]
        C, H, W = image.shape
        s = self.stride

        def _ceil_to_stride(x: int, s: int) -> int:
            return ((x + s - 1) // s) * s

        H_pad = _ceil_to_stride(H, s)
        W_pad = _ceil_to_stride(W, s)

        if H_pad == H and W_pad == W:
            # 已经是 stride 倍数，直接返回
            return image, raw_boxes

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

        # raw_boxes 坐标无需平移；但要裁剪到新边界（避免越界）
        if raw_boxes.shape[0] > 0:
            # [class_id, obj_id, x1, y1, x2, y2]
            # 保持 x1,y1,x2,y2 不变，但剪裁到 [0,W_pad] / [0,H_pad]
            raw_boxes[:, 2] = np.clip(raw_boxes[:, 2], 0, W_pad - 1)
            raw_boxes[:, 4] = np.clip(raw_boxes[:, 4], 0, W_pad)
            raw_boxes[:, 3] = np.clip(raw_boxes[:, 3], 0, H_pad - 1)
            raw_boxes[:, 5] = np.clip(raw_boxes[:, 5], 0, H_pad)

        return image, raw_boxes


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