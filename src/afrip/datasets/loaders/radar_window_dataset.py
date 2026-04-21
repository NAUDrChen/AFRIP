from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from afrip.datasets.registry import DATASETS

try:
    import scipy.io
except ImportError:
    scipy = None
try:
    import h5py
except ImportError:
    h5py = None
    
class LRUCacheMat:
    """
    简单 LRU 缓存: key=文件路径, value= ndarray data_raw
    """
    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> Optional[np.ndarray]:
        if path in self._data:
            self._data.move_to_end(path)
            return self._data[path]
        return None

    def put(self, path: str, arr: np.ndarray):
        self._data[path] = arr
        self._data.move_to_end(path)
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

def load_mat_data_raw(mat_path: str, var_name: str = "data_raw") -> np.ndarray:
    """
    兼容 v7.3 (HDF5) 与普通 mat。
    返回二维/二维复数矩阵: shape (azimuth, range) => (Y, X)
    若为复数矩阵，数据满足: A = A_real + j * A_imag
    """
    # 尝试 h5py
    if h5py is not None:
        try:
            with h5py.File(mat_path, "r") as f:
                if var_name in f:
                    dset = f[var_name]
                    # v7.3 复数: 复数元素 = real + j*imag
                    # 复数幅度: |z| = sqrt(real^2 + imag^2)
                    # 复数相位: phase(z) = atan2(imag, real)

                    if isinstance(dset.dtype, np.dtype) and dset.dtype.kind == 'V' and dset.dtype.names and set(dset.dtype.names) >= {"real", "imag"}:
                        real_part = dset["real"][()]
                        imag_part = dset["imag"][()]
                        arr = real_part + 1j * imag_part
                    else:
                        arr = dset[()]
                    # squeeze 去除多余维度 (如 (Y,X,1))
                    arr = np.squeeze(arr)
                    if arr.ndim != 2:
                        raise ValueError(f"{var_name} 期望二维矩阵，实际形状 {arr.shape}")
                    return arr
        except OSError:
            pass
        
    if scipy is not None:
        mat = scipy.io.loadmat(mat_path)
        if var_name not in mat:
            raise KeyError(f"{var_name} not found in {mat_path}")
        arr = np.squeeze(mat[var_name])
        if arr.ndim != 2:
            raise ValueError(f"{var_name} expects 2D matrix, got {arr.shape}")
        return arr

    raise RuntimeError("scipy or h5py is required to load mat files")


def boxes_csv_reader(csv_path: str) -> List[Dict[str, Any]]:
    """
    每行格式: [file name,class name,id,x1,y1,w,h,difficult] 与DarkLabel一致
    坐标假设: x1,y1 为左上角, w,h 为宽高 (单位与矩阵索引一致: x=列, y=行)
    """
    items: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            file_name, cls_name, obj_id, x1, y1, w, h, diff = row
            items.append(
                {
                    "file": file_name,
                    "class_name": cls_name,
                    "obj_id": int(obj_id),
                    "x1": float(x1),
                    "y1": float(y1),
                    "w": float(w),
                    "h": float(h),
                    "difficult": int(diff),
                }
            )
    return items

@DATASETS.register("RadarWindowDataset")
class RadarWindowDataset(Dataset):
    """
    将每个 mat 的 data_raw 按窗口切片。
    复数处理模式:
      stack: 输出 [2,H,W], 通道0=real, 通道1=imag
      magnitude_phase: 输出 [2,H,W], 通道0=|z|=sqrt(r^2+i^2), 通道1=phase=atan2(i,r)
      abs: 输出 [1,H,W], 通道0=|z|
    dB 转换使用公式: A_dB = 20 * log10(|A| + ε), ε>0 防止 log(0)
    YOLO 归一化:
      原始局部框 (x1,y1,x2,y2)
      中心: cx = (x1 + x2)/2, cy = (y1 + y2)/2
      尺寸: w = x2 - x1, h = y2 - y1
      归一化: cx_n = cx / W_win, cy_n = cy / H_win, w_n = w / W_win, h_n = h / H_win
    窗口覆盖策略: 在 stride 网格基础上补尾部窗口，确保最后区域被覆盖(允许 padding)。
    """
    def __init__(
        self,
        mat_dir: str,
        csv_path: str,
        window_size: Tuple[int, int] = (512, 512),
        stride: Tuple[int, int] = (256, 256),
        var_name: str = "data_raw",
        padding_value: float = 0.0,
        complex_mode: str = "abs",
        class_mapping: Optional[Dict[str, int]] = None,
        min_box_area: float = 4.0,
        cache_mat_files: int = 4,
        filter_difficult: bool = False,
        transforms=None,
        subset: Optional[str] = None,
        azimuth_split_ratio: float = 0.7,
        full_frame: bool = False
    ):
        self.mat_dir = mat_dir
        self.csv_path = csv_path
        self.var_name = var_name
        self.window_h, self.window_w = window_size
        self.stride_h, self.stride_w = stride
        self.padding_value = padding_value
        self.complex_mode = complex_mode
        self.class_mapping = class_mapping or {}
        self.min_box_area = min_box_area
        self.filter_difficult = filter_difficult
        self.cache = LRUCacheMat(cache_mat_files)
        self.transforms = transforms
        self.subset = subset  # 'train' / 'test' / None
        self.azimuth_split_ratio = azimuth_split_ratio
        self.full_frame = full_frame

        if self.full_frame:
            self.window_h, self.window_w = -1, -1

        # 读取标注
        ann_items = boxes_csv_reader(csv_path)
        # 按文件聚合
        self.file_to_boxes: Dict[str, List[Dict[str, Any]]] = {}
        for it in ann_items:
            if self.filter_difficult and it["difficult"] == 1:
                continue
            self.file_to_boxes.setdefault(it["file"], []).append(it)

        # 收集全部 mat 文件（只添加存在标注的文件；如需全部，可改）
        self.files = sorted(list(self.file_to_boxes.keys()))
        if not self.files:
            raise RuntimeError("未找到任何标注对应的文件")

        # 预构建窗口索引: [(file, y0, x0)]，file_name: 原始 .mat 文件名，y0: 窗口在原始矩阵中的起始行(纵向偏移)，x0: 窗口在原始矩阵中的起始列(横向偏移)
        self.index: List[Tuple[str, int, int]] = []
        self._build_index()

    def _index_cache_key(self) -> str:
        """
        生成影响窗口索引的哈希键：
        - 文件列表及其 (size, mtime)
        - 参数: window_size, stride, subset, azimuth_split_ratio
        注：索引仅与矩阵尺寸有关，不与标注或 complex_mode 等无关。
        """
        h = hashlib.sha256()
        # 参与的参数
        payload = {
            "window_size": (self.window_h, self.window_w),
            "stride": (self.stride_h, self.stride_w),
            "subset": self.subset,
            "azimuth_split_ratio": self.azimuth_split_ratio,
            "full_frame": self.full_frame,
            # 若未来索引逻辑依赖其他参数，可加入
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))

        # 文件列表及其 stat 信息
        for fname in self.files:
            fpath = os.path.join(self.mat_dir, fname)
            try:
                st = os.stat(fpath)
                # 使用 (size, mtime) 即可感知变更
                h.update(fname.encode("utf-8"))
                h.update(str(st.st_size).encode("utf-8"))
                h.update(str(int(st.st_mtime)).encode("utf-8"))
            except FileNotFoundError:
                # 若缺失视为不同键
                h.update((fname + ":missing").encode("utf-8"))

        return h.hexdigest()

    def _index_cache_path(self) -> str:
        cache_dir = os.path.join(self.mat_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        key = self._index_cache_key()
        return os.path.join(cache_dir, f"index_{key}.json")

    def _load_index_cache(self) -> Optional[List[Tuple[str, int, int]]]:
        path = self._index_cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 验证基本结构
            idx = data.get("index")
            files = data.get("files")
            if not isinstance(idx, list) or not isinstance(files, list):
                return None
            # 文件列表一致性检查
            if files != self.files:
                return None
            # 转换为期望类型
            out = []
            for item in idx:
                # item = [file, y0, x0]
                if not (isinstance(item, list) and len(item) == 3):
                    return None
                out.append((item[0], int(item[1]), int(item[2])))
            return out
        except Exception:
            return None

    def _save_index_cache(self, index_list: List[Tuple[str, int, int]]) -> None:
        path = self._index_cache_path()
        try:
            payload = {
                "files": self.files,
                "index": [[f, int(y0), int(x0)] for (f, y0, x0) in index_list],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            # 缓存失败不影响正常流程
            pass

    def _build_index(self):

        index: List[Tuple[str, int, int]] = []

        if self.full_frame:
            index = []
            for file in self.files:
                full = self._load_full_matrix(file)
                H, W = full.shape
                boundary_y = int(H * self.azimuth_split_ratio)
                if self.subset == 'train':
                    seg_y0 = 0
                    seg_h = max(boundary_y, 1)
                elif self.subset == 'test':
                    seg_y0 = boundary_y
                    seg_h = max(H - boundary_y, 1)
                else:
                    seg_y0 = 0
                    seg_h = H
                # 更新当前窗口尺寸为该文件段的真实尺寸
                if seg_h > self.window_h:
                    self.window_h = seg_h
                self.window_w = W
                index.append((file, seg_y0, 0))
                # 释放缓存对象，保持轻量
                self.cache = LRUCacheMat(self.cache.max_size)
            self.index = index
            self._save_index_cache(self.index)
            return
        
        # 先尝试加载缓存,不适用整帧模式
        cached = self._load_index_cache()
        if cached is not None:
            self.index = cached
            # 保持缓存对象，但清空已加载矩阵（索引不需要矩阵）
            self.cache = LRUCacheMat(self.cache.max_size)
            return

        # 未命中缓存，则正常构建
        for file in self.files:
            # 仅为获取 H,W 需要读取一次矩阵的形状；为了减少内存压力，不保留整阵。
            # 使用已有加载函数，但不持久缓存该矩阵（随即清空 LRU 以释放）
            full = self._load_full_matrix(file)
            H, W = full.shape  # Y, X

            y_positions = list(range(0, max(H - self.window_h + 1, 0), self.stride_h))
            x_positions = list(range(0, max(W - self.window_w + 1, 0), self.stride_w))
            if not y_positions or y_positions[-1] + self.window_h < H:
                y_positions.append(max(H - self.window_h, 0))
            if not x_positions or x_positions[-1] + self.window_w < W:
                x_positions.append(max(W - self.window_w, 0))

            boundary_y = int(H * self.azimuth_split_ratio)
            for y0 in y_positions:
                center_y = y0 + self.window_h / 2.0
                if self.subset == 'train' and center_y >= boundary_y:
                    continue
                if self.subset == 'test' and center_y < boundary_y:
                    continue
                for x0 in x_positions:
                    index.append((file, y0, x0))

            # 释放当前矩阵，避免占用LRU空间
            self.cache = LRUCacheMat(self.cache.max_size)

        self.index = index
        # 保存缓存以便下次快速加载
        self._save_index_cache(self.index)

    def _load_full_matrix(self, file: str) -> np.ndarray:
        path = os.path.join(self.mat_dir, file)
        arr = self.cache.get(path)
        if arr is None:
            arr = load_mat_data_raw(path, self.var_name).transpose()
            # 保证二维
            if arr.ndim != 2:
                # 若是 (Y,X,1) 之类
                arr = np.squeeze(arr)
            self.cache.put(path, arr)
        return arr

    def __len__(self):
        return len(self.index)

    def _extract_window(self, full: np.ndarray, y0: int, x0: int) -> np.ndarray:
        H_full, W_full = full.shape
        y1 = y0 + self.window_h
        x1 = x0 + self.window_w
        # 需要 padding ?
        pad_bottom = max(0, y1 - H_full)
        pad_right = max(0, x1 - W_full)
        slice_h = self.window_h - pad_bottom
        slice_w = self.window_w - pad_right
        window = full[y0:y0+slice_h, x0:x0+slice_w]
        if pad_bottom > 0 or pad_right > 0:
            out = np.full((self.window_h, self.window_w), self.padding_value, dtype=window.dtype)
            out[:slice_h, :slice_w] = window
            window = out
        return window

    def _convert_complex(self, arr: np.ndarray) -> np.ndarray:
        """
        幅度 |z| = sqrt(real^2 + imag^2)
        相位 phase = atan2(imag, real)
        """
        if np.iscomplexobj(arr):
            real = arr.real.astype(np.float32)
            imag = arr.imag.astype(np.float32)
            if self.complex_mode == "stack":
                return np.stack([real, imag], axis=0)
            elif self.complex_mode == "magnitude_phase":
                # mag = sqrt(r^2 + i^2); phase = atan2(i, r)
                mag = np.sqrt(real**2 + imag**2)
                phase = np.arctan2(imag, real)
                return np.stack([mag.astype(np.float32), phase.astype(np.float32)], axis=0)
            elif self.complex_mode == "abs":
                # 单通道: |z| = sqrt(r^2 + i^2)
                mag = np.sqrt(real**2 + imag**2)
                return mag[None, ...].astype(np.float32)
            else:
                raise ValueError(f"未知 complex_mode: {self.complex_mode}")
        else:
            # 实数 => 单通道
            return arr.astype(np.float32)[None, ...]  # [1,H,W]

    def _boxes_in_window(
        self,
        boxes: List[Dict[str, Any]],
        y0: int,
        x0: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        窗口内目标筛选:
        交集计算:
          原框: [x1,y1,x2,y2]
          窗口: [x0,y0,x0+W_win,y0+H_win]
          交集: inter_x1 = max(x1, win_x1)
                inter_y1 = max(y1, win_y1)
                inter_x2 = min(x2, win_x2)
                inter_y2 = min(y2, win_y2)
          若 inter_x2 <= inter_x1 或 inter_y2 <= inter_y1 => 无交集
          面积: inter_w * inter_h < self.min_box_area 过滤最小面积
        YOLO 坐标:
          cx = (local_x1 + local_x2)/2
          cy = (local_y1 + local_y2)/2
          bw = local_x2 - local_x1
          bh = local_y2 - local_y1
          归一化: 除以 (window_w, window_h)
        """
        out_raw = []
        out_yolo = []
        for b in boxes:
            x1 = b["x1"]
            y1 = b["y1"]
            w = b["w"]
            h = b["h"]
            x2 = x1 + w
            y2 = y1 + h
            # 与窗口范围求交
            win_x1, win_y1 = x0, y0
            win_x2, win_y2 = x0 + self.window_w, y0 + self.window_h

            inter_x1 = max(x1, win_x1)
            inter_y1 = max(y1, win_y1)
            inter_x2 = min(x2, win_x2)
            inter_y2 = min(y2, win_y2)
            # 交集宽高: inter_w = inter_x2 - inter_x1; inter_h = inter_y2 - inter_y1
            # 过滤面积: inter_w * inter_h >= min_box_area
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue  # 无交集

            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            if inter_w * inter_h < self.min_box_area:
                continue

            # 映射到窗口局部坐标
            local_x1 = inter_x1 - x0
            local_y1 = inter_y1 - y0
            local_x2 = inter_x2 - x0
            local_y2 = inter_y2 - y0

            class_name = b["class_name"]
            class_id = self.class_mapping.get(class_name, 0)  # 默认0
            obj_id = b["obj_id"]

            out_raw.append([class_id, obj_id, local_x1, local_y1, local_x2, local_y2])

            # 转 YOLO 格式 (cx,cy,w,h) 归一化
            cx = (local_x1 + local_x2) / 2.0  # (x1+x2)/2
            cy = (local_y1 + local_y2) / 2.0  # (y1+y2)/2
            bw = local_x2 - local_x1          # w = x2 - x1
            bh = local_y2 - local_y1          # h = y2 - y1
            # 归一化: / window 尺寸
            cx_n = cx / self.window_w
            cy_n = cy / self.window_h
            bw_n = bw / self.window_w
            bh_n = bh / self.window_h
            out_yolo.append([class_id, obj_id, cx_n, cy_n, bw_n, bh_n])

        if out_raw:
            return (
                np.array(out_raw, dtype=np.float32),
                np.array(out_yolo, dtype=np.float32)
            )
        else:
            return (
                np.zeros((0, 6), dtype=np.float32),
                np.zeros((0, 6), dtype=np.float32)
            )

    def _raw_to_yolo(self, raw_boxes: np.ndarray) -> np.ndarray:
        # raw_boxes: [N,6] => [class_id, obj_id, x1, y1, x2, y2]
        if raw_boxes.shape[0] == 0:
            return np.zeros((0,6), dtype=np.float32)
        out = []
        for r in raw_boxes:
            class_id, obj_id, x1, y1, x2, y2 = r
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = (x2 - x1)
            h = (y2 - y1)
            cx_n = cx / self.window_w
            cy_n = cy / self.window_h
            w_n = w / self.window_w
            h_n = h / self.window_h
            out.append([class_id, obj_id, cx_n, cy_n, w_n, h_n])
        return np.array(out, dtype=np.float32)

    def __getitem__(self, idx: int):
        file, y0, x0 = self.index[idx]
        full = self._load_full_matrix(file)
        window = self._extract_window(full, y0, x0)  # (H,W)
        tensor = self._convert_complex(window)  # [C,H,W]

        boxes = self.file_to_boxes.get(file, [])
        raw_boxes, yolo_boxes = self._boxes_in_window(boxes, y0, x0)

        # 应用 transform (若存在)
        if self.transforms is not None:
            tensor_t = torch.from_numpy(tensor)  # 先转 tensor
            tensor_t, raw_boxes = self.transforms(tensor_t, raw_boxes)
            # 重新生成 yolo
            yolo_boxes = self._raw_to_yolo(raw_boxes)
            # 保持 raw_boxes, yolo_boxes 为 numpy 格式后再转 torch
            tensor = tensor_t.numpy()
        else:
            yolo_boxes = self._raw_to_yolo(raw_boxes)
            
        # 构造 targets 为 List[Dict]，单元素：{'boxes': [N,4], 'labels': [N,]}
        if raw_boxes.shape[0] > 0:
            tgt_boxes = torch.from_numpy(yolo_boxes[:, 2:6]).float()      # [cx, cy, w, h]
            tgt_labels = torch.from_numpy(raw_boxes[:, 0]).long()        # class_id
        else:
            tgt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            tgt_labels = torch.zeros((0,), dtype=torch.long)
        targets_list = [{"boxes": tgt_boxes, "labels": tgt_labels}]

        sample = {
            "image": torch.from_numpy(tensor),
            "targets": targets_list,                # List[Dict]
            "raw_boxes": torch.from_numpy(raw_boxes),
            "meta": {
                "file": file,
                "global_origin": (y0, x0),
                "index": idx
            }
        }
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)

        # __getitem__ 中 targets 始终是长度为 1 的列表，取其第 0 个 Dict
        targets = [b["targets"][0] for b in batch]

        raw_boxes_list = []
        for bi, b in enumerate(batch):
            rb = b["raw_boxes"]  # [N,6] 或空
            if rb.numel() > 0:
                bi_col = torch.full((rb.shape[0], 1), bi, dtype=rb.dtype)
                raw_boxes_list.append(torch.cat([bi_col, rb], dim=1))  # [N,7]

        if raw_boxes_list:
            raw_boxes = torch.cat(raw_boxes_list, dim=0)
        else:
            raw_boxes = torch.zeros((0, 7), dtype=torch.float32)

        return {
            "images": images,              # [B,C,H,W]
            "targets": targets,            # List[Dict{boxes:[Ni,4], labels:[Ni]}]
            "raw_boxes": raw_boxes,        # [M,7]: (batch_index,class_id,obj_id,x1,y1,x2,y2)
            "batch_meta": [b["meta"] for b in batch]
        }