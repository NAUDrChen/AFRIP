from typing import List, Tuple, Dict, Any, Optional
import os
import matplotlib
if os.environ.get("AFRIP_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from afrip.utils.box_ops import rescale_bboxes
import numpy as np
import re
import math
from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset
import cv2

def compute_roc_points(pr_curve: Dict[str, Any],
                                             total_gt: int,
                                             total_non_target_cells: int) -> Tuple[np.ndarray, np.ndarray]:
        """由 PR 曲线恢复 ROC 曲线点。

        返回:
            - fpr: False Positive Rate
            - tpr: True Positive Rate
        """
        rec = np.array(pr_curve.get("rec", []), dtype=float)
        prec = np.array(pr_curve.get("prec", []), dtype=float)
        if rec.size == 0 or prec.size == 0:
                return np.array([], dtype=float), np.array([], dtype=float)

        safe_prec = np.clip(prec, 1e-12, 1.0)
        safe_total_non_target = max(int(total_non_target_cells), 1)
        safe_total_gt = max(int(total_gt), 1)

        fp = (1.0 / safe_prec - 1.0) * rec * safe_total_gt
        fpr = fp / safe_total_non_target
        tpr = rec
        return fpr, tpr


def plot_roc_curve(pr_curve: Dict[str, Any],
                                     total_gt: int,
                                     total_non_target_cells: int,
                                     epoch: int,
                                     save_dir: str = "results",
                                     label: str = "ROC",
                                     ax=None,
                                     save_path: Optional[str] = None):
    """
    基于 PR 曲线数据绘制 ROC：
      - TPR = recall
      - FPR 由 precision/recall 反推 FP，再除以非目标单元数
    文件命名按轮次：roc_epoch_{epoch+1:03d}.png
    """
    import os
    import matplotlib.pyplot as plt

    fpr, tpr = compute_roc_points(pr_curve, total_gt, total_non_target_cells)
    if fpr.size == 0 or tpr.size == 0:
        return {
            "fpr": fpr,
            "tpr": tpr,
            "out_path": None,
        }

    created_fig = False
    if ax is None:
        created_fig = True
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.semilogx(fpr, tpr, label=label)
    existing_labels = set(ax.get_legend_handles_labels()[1])
    if "Chance" not in existing_labels:
        ax.plot([1e-6, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve (epoch {epoch+1})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = save_path
    if created_fig:
        if out_path is None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"roc_epoch_{epoch+1:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "out_path": out_path,
    }

def visualize_full_predictions(dataset: RadarWindowDataset,
                               records: List[Dict[str, Any]],
                               conf_thresh: float = 0.25,
                               iou_thresh: float = 0.5,
                               max_files: int = 20,
                               epoch: Optional[int] = None):
    """
    使用评估阶段缓存的 prediction_records 在整幅距离-方位图上叠加：
      - 预测框 (score>=conf_thresh)
      - GT 框
      - 简单 TP/FP 标记（基于与任意 GT IoU>=iou_thresh）
    无需重新模型前向。
    records: [
      {
        'file': str,
        'origin': (y0, x0),
        'pred_boxes': Tensor[P,4] (窗口内 xy    xy),
        'pred_scores': Tensor[P],
        'gt_boxes': Tensor[G,4] (窗口内 xyxy)
      }, ...
    ]
    """
    import torch
    from afrip.utils.box_ops import box_iou
    # 按文件聚合
    file_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        file_groups.setdefault(r["file"], []).append(r)
    shown = 0
    eps = 1e-12
    save_dir = os.path.join("results", "eval_full")
    os.makedirs(save_dir, exist_ok=True)
    for file_name, rec_list in file_groups.items():
        if shown >= max_files:
            break
        full_mat = dataset._load_full_matrix(file_name)
        if np.iscomplexobj(full_mat):
            amp_full = np.abs(full_mat)
        else:
            amp_full = np.abs(full_mat.astype(np.float32))
        db_full = 20.0 * np.log10(amp_full + eps)
        v1, v2 = np.percentile(db_full, [1, 99])
        disp = np.clip((db_full - v1) / (v2 - v1 + 1e-6), 0, 1)

        # plt.figure(figsize=(7,6))
        # ax = plt.gca()
        # ax.imshow(disp, cmap='viridis', origin='upper')
        # ax.set_title(f"Full Predictions: {file_name}")
        # ax.axis('off')
        H_full, W_full = amp_full.shape
        target_dpi = 100  # 可调；最终像素 = figsize * dpi = (W_full, H_full)
        fig = plt.figure(figsize=(W_full / target_dpi, H_full / target_dpi), dpi=target_dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(111)
        ax.imshow(disp, cmap='viridis', origin='upper')
        ax.set_title(f"Full Predictions: {file_name}", fontsize=10)
        ax.axis('off')
        # 先收集全局 GT
        global_gt = []
        for rec in rec_list:
            y0, x0 = rec["origin"]
            for g in rec["gt_boxes"]:
                x1,y1,x2,y2 = g.tolist()
                global_gt.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0])
        if global_gt:
            global_gt_t = torch.tensor(global_gt, dtype=torch.float32)
        else:
            global_gt_t = torch.zeros((0,4), dtype=torch.float32)

        # 绘制 GT
        for (x1,y1,x2,y2) in global_gt:
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                             edgecolor='lime', facecolor='none', linewidth=1.0)
            ax.add_patch(rect)

        # 绘制预测并标记 TP/FP
        for rec in rec_list:
            y0, x0 = rec["origin"]
            pb = rec["pred_boxes"]
            ps = rec["pred_scores"]
            if pb.numel() == 0:
                continue
            # 过滤阈值
            keep = ps >= conf_thresh
            pb = pb[keep]
            ps = ps[keep]
            if pb.numel() == 0:
                continue
            # IoU 与 TP 判定
            if global_gt_t.numel() > 0:
                ious, _ = box_iou(pb, global_gt_t)  # [P, G]
                max_iou, _ = ious.max(dim=1)
                is_tp = max_iou >= iou_thresh
            else:
                is_tp = torch.zeros((pb.shape[0],), dtype=torch.bool)
            for i in range(pb.shape[0]):
                x1,y1,x2,y2 = pb[i].tolist()
                gx1 = x1 + x0; gy1 = y1 + y0; gx2 = x2 + x0; gy2 = y2 + y0
                color = 'orange' if is_tp[i] else 'red'
                rect = Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                 edgecolor=color, facecolor='none', linewidth=1)
                ax.add_patch(rect)
                ax.text(gx1, max(0, gy1 - 6),
                        f"{ps[i]:.2f}{' T' if is_tp[i] else ' F'}",
                        color=color, fontsize=7, backgroundcolor='black')

        # 图例
        handles = [
            Rectangle((0,0),1,1, edgecolor='lime', facecolor='none', label='GT'),
            Rectangle((0,0),1,1, edgecolor='orange', facecolor='none', label='Pred TP'),
            Rectangle((0,0),1,1, edgecolor='red', facecolor='none', label='Pred FP'),
        ]
        ax.legend(handles=handles, loc='lower right', fontsize=8)
        # plt.tight_layout()
        # out_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_full_{shown+1}.png")
        # plt.savefig(out_path, dpi=150, bbox_inches='tight')
        # plt.close()
        out_base = os.path.splitext(file_name)[0]
        suffix = f"e{epoch}_" if epoch is not None else ""
        out_path = os.path.join(save_dir, f"{suffix}{out_base}_full_{shown+1}.png")
        fig.savefig(out_path, dpi=target_dpi)  # 输出尺寸应为 (W_full, H_full)
        plt.close(fig)
        shown += 1
def visualize_batch_with_full(dataset: RadarWindowDataset,
                              batch: Dict[str, Any],
                              complex_mode: str = "abs",
                              max_per_batch: int = 8,
                              image_is_normalized=True):
    """
    dB 转换: dB = 20 * log10(|A| + ε), ε ~ 1e-12 防止 log(0)
    归一显示: norm = clip((dB - p1)/(p99 - p1), 0, 1) 其中 p1,p99 为分位数增强对比度
    """
    images = batch["images"]      # [B,C,H,W]
    raw_boxes = batch["raw_boxes"]  # [M,8]
    metas = batch["batch_meta"]
    B, C, H, W = images.shape

    file_groups: Dict[str, List[int]] = {}
    for i, m in enumerate(metas):
        file_groups.setdefault(m["file"], []).append(i)

    per_sample_boxes = [[] for _ in range(B)]
    for rb in raw_boxes:
        bi = int(rb[0].item())
        per_sample_boxes[bi].append(rb[1:].cpu().numpy())

    eps = 1e-12  # 防 log(0)

    for file_name, sample_indices in file_groups.items():
        full_mat = dataset._load_full_matrix(file_name)
        if np.iscomplexobj(full_mat):
            amp_full = np.abs(full_mat)
        else:
            amp_full = np.abs(full_mat.astype(np.float32))
        db_full = 20.0 * np.log10(amp_full + eps)  # 公式: 20 log10(|A| + ε)
        vmin_f, vmax_f = np.percentile(db_full, [1, 99])
        disp_full_norm = np.clip((db_full - vmin_f) / (vmax_f - vmin_f + 1e-6), 0, 1)

        show_indices = sample_indices[:max_per_batch]
        n_windows = len(show_indices)
        fig, axes = plt.subplots(1, n_windows + 1, figsize=(4*(n_windows + 1), 4))
        if isinstance(axes, np.ndarray):
            ax_full = axes[0]
            window_axes = axes[1:]
        else:
            ax_full = axes
            window_axes = []

        ax_full.imshow(disp_full_norm, cmap='viridis', origin='upper')
        ax_full.set_title(f"Full(dB): {file_name}\nRange[{vmin_f:.1f},{vmax_f:.1f}]")
        ax_full.set_axis_off()

        for si in show_indices:
            m = metas[si]
            y0, x0 = m["global_origin"]
            # 每样本实际窗口尺寸来自张量形状 [C,H,W]
            H_win = int(images[si].shape[-2])
            W_win = int(images[si].shape[-1])
            rect = Rectangle((x0, y0),
                             W_win, H_win,
                             edgecolor='red', facecolor='none', linewidth=1.0)
            ax_full.add_patch(rect)
            ax_full.text(x0 + 2, y0 + 14, f"{si}", color='yellow',
                         fontsize=8, backgroundcolor='black')

        for ax, si in zip(window_axes, show_indices):
            img_t = images[si].cpu().numpy()
            if complex_mode == "stack" and C == 2:
                amp = np.sqrt(img_t[0]**2 + img_t[1]**2)
            elif complex_mode == "magnitude_phase" and C == 2:
                # 第 0 通道假设是幅度
                amp = img_t[0]
            else:
                amp = img_t[0]
            amp = np.abs(amp)
            if image_is_normalized:
                disp_w = np.clip(amp, 0, 1)          # 已归一化，直接显示
            else:
                db = 20.0 * np.log10(amp + eps)      # 原始幅度，做 dB 变换
                vmin_w, vmax_w = np.percentile(db, [1, 99])
                disp_w = np.clip((db - vmin_w) / (vmax_w - vmin_w + 1e-6), 0, 1)

            ax.imshow(disp_w, cmap='viridis', origin='upper')
            if image_is_normalized:
                ax.set_title(f"Slice #{si}\n[normalized]")
            else:
                ax.set_title(f"Slice #{si}\ndB[{vmin_w:.1f},{vmax_w:.1f}]")
            ax.set_axis_off()

            for box in per_sample_boxes[si]:
                class_id, obj_id, x1, y1, x2, y2 = box
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 edgecolor='lime', facecolor='none', linewidth=1)
                ax.add_patch(rect)
                ax.text(x1, max(0, y1 - 3),
                        f"c{int(class_id)} id{int(obj_id)}",
                        color='yellow', fontsize=7, backgroundcolor='black')

        plt.tight_layout()
        plt.show()

def compute_and_visualize_stats(dataset: RadarWindowDataset):
    """
    增强版统计:
      文件级原始标注统计:
        - 类别计数
        - 宽/高/面积/长宽比分布
        - 归一化中心分布
        - 信杂比 SCR 分布
      新增:
        - 每个窗口中目标数量分布 (按窗口滑动后，窗口内满足最小交集面积的目标数量)
    """
    from collections import Counter
    class_counts = Counter()
    widths, heights, areas, aspect_ratios = [], [], [], []
    centers_x, centers_y = [], []
    scr_db = []
    scr_per_class: Dict[str, List[float]] = {}

    # 新增: 每窗口目标数量分布
    window_obj_counts: List[int] = []

    eps = 1e-12
    clutter_expand: float = 1.5
    min_clutter_pixels: int = 20
    visualize: bool = True

    # ========= 先统计窗口目标数量分布 =========
    win_h = dataset.window_h
    win_w = dataset.window_w
    min_area = dataset.min_box_area

    # 遍历所有窗口索引 (file, y0, x0)
    # ========= 先统计窗口目标数量分布 =========
    min_area = dataset.min_box_area

    for file, y0, x0 in dataset.index:
        # 动态窗口尺寸
        if getattr(dataset, "full_frame", False):
            full = dataset._load_full_matrix(file)
            H, W = full.shape
            boundary_y = int(H * dataset.azimuth_split_ratio)
            if dataset.subset == 'train':
                win_h = max(boundary_y, 1)
                win_w = W
            elif dataset.subset == 'test':
                win_h = max(H - boundary_y, 1)
                win_w = W
            else:
                win_h = H
                win_w = W
        else:
            win_h = dataset.window_h
            win_w = dataset.window_w

        boxes = dataset.file_to_boxes.get(file, [])
        count_in_window = 0
        win_x1, win_y1 = x0, y0
        win_x2, win_y2 = x0 + win_w, y0 + win_h
        for b in boxes:
            x1 = b["x1"]; y1b = b["y1"]; w = b["w"]; h = b["h"]
            x2 = x1 + w; y2b = y1b + h
            inter_x1 = max(x1, win_x1)
            inter_y1 = max(y1b, win_y1)
            inter_x2 = min(x2, win_x2)
            inter_y2 = min(y2b, win_y2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            if inter_w * inter_h < min_area:
                continue
            count_in_window += 1
        window_obj_counts.append(count_in_window)

    # ========= 原始文件级逐文件统计 =========
    for file, boxes in dataset.file_to_boxes.items():
        full = dataset._load_full_matrix(file)
        H, W = full.shape
        if np.iscomplexobj(full):
            amp = np.abs(full)
        else:
            amp = np.abs(full.astype(np.float32))
        power = (amp ** 2).astype(np.float32)

        targets_mask = np.zeros((H, W), dtype=bool)
        int_boxes = []
        for b in boxes:
            x1 = int(round(b["x1"]))
            y1 = int(round(b["y1"]))
            w_box = int(round(b["w"]))
            h_box = int(round(b["h"]))
            x2 = min(W, x1 + w_box)
            y2 = min(H, y1 + h_box)
            x1 = max(0, x1); y1 = max(0, y1)
            if x2 <= x1 or y2 <= y1:
                int_boxes.append((None, None, None, None))
                continue
            targets_mask[y1:y2, x1:x2] = True
            int_boxes.append((x1, y1, x2, y2))
        global_clutter_mask = ~targets_mask

        for idx_box, b in enumerate(boxes):
            x1_f = b["x1"]; y1_f = b["y1"]; w_f = b["w"]; h_f = b["h"]
            x1, y1, x2, y2 = int_boxes[idx_box]
            if x1 is None:
                continue
            widths.append(w_f)
            heights.append(h_f)
            areas.append(w_f * h_f)
            aspect_ratios.append(w_f / (h_f + 1e-6))
            cx = x1_f + w_f / 2.0
            cy = y1_f + h_f / 2.0
            centers_x.append(cx / W)
            centers_y.append(cy / H)
            class_counts[b["class_name"]] += 1

            obj_region = power[y1:y2, x1:x2]
            if obj_region.size == 0:
                continue
            obj_mean = float(obj_region.mean())

            cx_pix = (x1 + x2) / 2.0
            cy_pix = (y1 + y2) / 2.0
            w_ext = int(round((x2 - x1) * clutter_expand))
            h_ext = int(round((y2 - y1) * clutter_expand))
            x1e = int(round(cx_pix - w_ext / 2.0))
            y1e = int(round(cy_pix - h_ext / 2.0))
            x2e = x1e + w_ext
            y2e = y1e + h_ext
            x1e = max(0, x1e); y1e = max(0, y1e)
            x2e = min(W, x2e); y2e = min(H, y2e)
            if x2e <= x1e or y2e <= y1e:
                clutter_pixels = power[global_clutter_mask]
            else:
                ext_mask = np.zeros((H, W), dtype=bool)
                ext_mask[y1e:y2e, x1e:x2e] = True
                clutter_region_mask = ext_mask & (~targets_mask)
                clutter_pixels = power[clutter_region_mask]
                if clutter_pixels.size < min_clutter_pixels:
                    clutter_pixels = power[global_clutter_mask]

            if clutter_pixels.size == 0:
                continue
            clutter_mean = float(clutter_pixels.mean())
            scr = 10.0 * math.log10(obj_mean / (clutter_mean + eps))
            scr_db.append(scr)
            scr_per_class.setdefault(b["class_name"], []).append(scr)

    total_objs = sum(class_counts.values())
    print("====== 标注总体统计 (原始文件级) ======")
    print(f"目标总数: {total_objs}")
    for cls, cnt in class_counts.items():
        print(f"  类别 {cls}: {cnt} ({cnt/total_objs*100:.2f}%)")
    if widths:
        print(f"宽度均值/中位数: {np.mean(widths):.2f} / {np.median(widths):.2f}")
        print(f"高度均值/中位数: {np.mean(heights):.2f} / {np.median(heights):.2f}")
        print(f"面积均值/中位数: {np.mean(areas):.2f} / {np.median(areas):.2f}")
        print(f"长宽比均值/中位数: {np.mean(aspect_ratios):.3f} / {np.median(aspect_ratios):.3f}")
    if scr_db:
        print(f"SCR(dB) 均值/中位数: {np.mean(scr_db):.2f} / {np.median(scr_db):.2f}")
        print(f"SCR(dB) 最小/最大: {np.min(scr_db):.2f} / {np.max(scr_db):.2f}")

    # 新增窗口统计输出
    if window_obj_counts:
        zero_windows = sum(c == 0 for c in window_obj_counts)
        print("====== 窗口级目标数量统计 ======")
        print(f"窗口总数: {len(window_obj_counts)}")
        print(f"空窗口数量: {zero_windows} ({zero_windows/len(window_obj_counts)*100:.2f}%)")
        print(f"每窗口目标数均值/中位数: {np.mean(window_obj_counts):.3f} / {np.median(window_obj_counts):.3f}")
        print(f"最大目标数/标准差: {np.max(window_obj_counts)} / {np.std(window_obj_counts):.3f}")

    if visualize:
        # 原有分布图
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ax_bar = axes[0,0]; ax_w = axes[0,1]; ax_h = axes[0,2]
        ax_area = axes[1,0]; ax_ar = axes[1,1]; ax_center = axes[1,2]

        ax_bar.bar(list(class_counts.keys()), list(class_counts.values()), color='steelblue')
        ax_bar.set_title("类别数量分布"); ax_bar.set_ylabel("数量"); ax_bar.set_xlabel("类别")
        ax_bar.tick_params(axis='x', rotation=30)

        ax_w.hist(widths, bins=30, color='orange', alpha=0.8)
        ax_w.set_title("宽度分布"); ax_w.set_xlabel("width"); ax_w.set_ylabel("count")

        ax_h.hist(heights, bins=30, color='green', alpha=0.8)
        ax_h.set_title("高度分布"); ax_h.set_xlabel("height"); ax_h.set_ylabel("count")

        ax_area.hist(areas, bins=30, color='purple', alpha=0.8)
        ax_area.set_title("面积分布"); ax_area.set_xlabel("area"); ax_area.set_ylabel("count")
        ax_area.set_yscale('log')

        ax_ar.hist(aspect_ratios, bins=30, color='red', alpha=0.8)
        ax_ar.set_title("长宽比 w/h 分布"); ax_ar.set_xlabel("aspect ratio"); ax_ar.set_ylabel("count")

        hb = ax_center.hexbin(centers_x, centers_y, gridsize=30, cmap='viridis')
        ax_center.set_title("归一化中心分布"); ax_center.set_xlabel("cx_norm"); ax_center.set_ylabel("cy_norm")
        ax_center.invert_yaxis()
        fig.colorbar(hb, ax=ax_center, shrink=0.75, label='count')
        plt.tight_layout()
        plt.show()

        # 新增窗口目标数量分布可视化
        if window_obj_counts:
            from collections import Counter
            freq = Counter(window_obj_counts)
            sorted_items = sorted(freq.items())
            counts_x = [k for k, _ in sorted_items]
            counts_y = [v for _, v in sorted_items]
            total_windows = len(window_obj_counts)
            # 计算累计分布
            cumulative = []
            running = 0
            for v in counts_y:
                running += v
                cumulative.append(running / total_windows)

            plt.figure(figsize=(8,4))
            ax1 = plt.gca()
            ax1.bar(counts_x, counts_y, color='slateblue', alpha=0.85, label='频次')
            ax1.set_xlabel("窗口内目标数量")
            ax1.set_ylabel("窗口计数")
            ax1.grid(alpha=0.3, axis='y')

            ax2 = ax1.twinx()
            ax2.plot(counts_x, cumulative, color='darkred', marker='o', linewidth=1.5, label='累计比例(CDF)')
            ax2.set_ylabel("累计比例")
            ax2.set_ylim(0, 1.05)

            # 合并图例
            lines_labels = []
            for ax in [ax1, ax2]:
                handles, labels = ax.get_legend_handles_labels()
                lines_labels.extend(zip(handles, labels))
            unique = dict(lines_labels)
            ax1.legend(unique.keys(), unique.values(), loc='upper right', fontsize=9)

            plt.title("每窗口目标数量分布 (频次 + 累计)")
            plt.tight_layout()
            plt.show()

        # SCR 可视化
        if scr_db:
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.hist(scr_db, bins=40, color='teal', alpha=0.85)
            plt.title("SCR(dB) 总体分布")
            plt.xlabel("SCR (dB)")
            plt.ylabel("count")
            plt.grid(alpha=0.3)

            plt.subplot(1,2,2)
            for cls, vals in scr_per_class.items():
                if vals:
                    plt.hist(vals, bins=40, alpha=0.5, label=cls)
            plt.title("SCR(dB) 按类别分布")
            plt.xlabel("SCR (dB)")
            plt.ylabel("count")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6,5))
            plt.scatter(areas, scr_db, s=12, alpha=0.5)
            plt.title("面积 vs SCR(dB)")
            plt.xlabel("area")
            plt.ylabel("SCR(dB)")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return {
        "class_counts": class_counts,
        "total": total_objs,
        "widths": widths,
        "heights": heights,
        "areas": areas,
        "aspect_ratios": aspect_ratios,
        "scr_db": scr_db,
        "scr_per_class": scr_per_class,
        "window_obj_counts": window_obj_counts
    }

def visualize_single_frame_dets_and_gt(
    images,
    pred_bboxes,
    pred_scores,
    raw_boxes,
    conf_thresh: float = 0.25,
    complex_mode: str = "abs",
    save_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
    dpi: int = 100
):
    """
    在单帧图像上可视化检测结果与GT。
    - images: Tensor 或 np.ndarray，形状支持 [B,C,H,W] 或 [C,H,W]；已做 AmplitudeNormalize + PadToStride
    - pred_bboxes: Tensor/np.ndarray [N,4]，绝对坐标 xyxy
    - pred_scores: Tensor/np.ndarray [N]
    - targets: Dict，含 'boxes' [M,4] 归一化坐标 cxcywh 或 xyxy? 本项目采用 yolo 归一化的 xyxy，请按当前实现读取
      若为归一化 xyxy，则按当前图像尺寸反归一化；若为归一化 cxcywh 则先转回 xyxy
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # 统一张量为CPU numpy
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    imgs_np = to_numpy(images)
    preds_np = to_numpy(pred_bboxes)
    scores_np = to_numpy(pred_scores)

    # 取单帧 [C,H,W]
    if imgs_np.ndim == 4:  # [B,C,H,W]
        img_np = imgs_np[0]
    elif imgs_np.ndim == 3:  # [C,H,W]
        img_np = imgs_np
    else:
        raise ValueError(f"images 维度不支持: {imgs_np.shape}")

    C, H, W = img_np.shape

    # 显示底图：AmplitudeNormalize 后本身在 [0,1]；若为复栈，取幅度显示
    if C == 2:
        real = img_np[0]
        imag = img_np[1]
        disp = np.sqrt(real ** 2 + imag ** 2)
    else:
        disp = img_np[0]

    # 组装检测框（过滤阈值）
    if preds_np is None or preds_np.size == 0:
        preds_np = np.zeros((0, 4), dtype=np.float32)
        scores_np = np.zeros((0,), dtype=np.float32)
    else:
        # 按阈值过滤
        keep = scores_np >= float(conf_thresh)
        preds_np = preds_np[keep]
        scores_np = scores_np[keep]

    # 绘图
    target_dpi = dpi
    fig = plt.figure(figsize=(W / target_dpi, H / target_dpi), dpi=target_dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.imshow(disp, cmap='viridis', origin='upper')
    ax.axis('off')
    ax.set_title("Detections vs GT", fontsize=10)

    gt_boxes_abs=to_numpy(raw_boxes)
    gt_labels=gt_boxes_abs[:,1]
    # 绘制 GT：绿色
    for i in range(raw_boxes.shape[0]):
        x1, y1, x2, y2 = gt_boxes_abs[i][3:]
        w_box = max(0.0, x2 - x1)
        h_box = max(0.0, y2 - y1)
        rect = Rectangle((x1, y1), w_box, h_box, linewidth=1.0, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 1, f"GT:{int(gt_labels[i])}", color='lime', fontsize=8, va='bottom')

    # 绘制检测：红色，带 score
    for i in range(preds_np.shape[0]):
        x1, y1, x2, y2 = preds_np[i]
        w_box = max(0.0, x2 - x1)
        h_box = max(0.0, y2 - y1)
        rect = Rectangle((x1, y1), w_box, h_box, linewidth=1.0, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        sc = float(scores_np[i]) if i < len(scores_np) else 0.0
        ax.text(x1, y1 - 1, f"Det:{sc:.2f}", color='red', fontsize=8, va='bottom')

    # 保存或显示
    if save_path is None and save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "single_frame_dets_gt.png")

    if save_path is not None:
        fig.savefig(save_path, dpi=target_dpi)
    if show:
        plt.show()
    plt.close(fig)
   

# 模块级: 管理多个视频写入器（按路径复用）
_VIDEO_WRITERS: Dict[str, cv2.VideoWriter] = {}

def _get_or_create_vwriter(path: str, width: int, height: int, fps: int = 20, codec: str = "mp4v"):
    """
    获取或创建 cv2.VideoWriter。width/height 为像素尺寸。
    codec 示例: 'mp4v' (MP4), 'XVID' (AVI)
    """
    global _VIDEO_WRITERS
    if path in _VIDEO_WRITERS:
        return _VIDEO_WRITERS[path]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # 注意: OpenCV 期望 (width, height)
    vw = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
    if not vw or not vw.isOpened():
        raise RuntimeError(f"无法打开视频写入器: {path}")
    _VIDEO_WRITERS[path] = vw
    return vw

def close_all_video_writers():
    """关闭并清理所有视频写入器，确保文件落盘。"""
    global _VIDEO_WRITERS
    for p, vw in list(_VIDEO_WRITERS.items()):
        try:
            vw.release()
        except Exception:
            pass
    _VIDEO_WRITERS.clear()

def visualize_tracks_on_image(
    images,
    tracks_xyxy_cls_id,
    track_histories=None,
    conf_thresh: float = 0.25,
    complex_mode: str = "abs",
    save_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
    dpi: int = 100,
    # 新增参数：视频保存
    video_path: Optional[str] = None,
    video_fps: int = 20,
    video_codec: str = "mp4v"  # 常用: 'mp4v' (mp4), 'XVID' (avi)
):
    """
    在单帧图像上可视化跟踪结果和轨迹。
    - images: Tensor/ndarray [B,C,H,W] 或 [C,H,W]
    - tracks_xyxy_cls_id: list/ndarray/torch.Tensor，形如 [[x1,y1,x2,y2,track_cls,track_id], ...]
    - track_histories: dict[int, List[Tuple[float,float]]]，每个 track_id 的中心点历史 [(cx,cy), ...]
    - 若提供 video_path，则将该帧写入对应视频。多次调用会持续追加到同一个文件。
      结束后请调用 close_all_video_writers() 以写盘并释放资源。
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    imgs_np = to_numpy(images)
    tracks_np = to_numpy(tracks_xyxy_cls_id)
    if tracks_np is None:
        tracks_np = np.zeros((0, 6), dtype=np.float32)

    # 单帧 [C,H,W]
    if imgs_np.ndim == 4:
        img_np = imgs_np[0]
    elif imgs_np.ndim == 3:
        img_np = imgs_np
    else:
        raise ValueError(f"images 维度不支持: {imgs_np.shape}")

    C, H, W = img_np.shape
    if C == 2:
        real = img_np[0]; imag = img_np[1]
        disp = np.sqrt(real**2 + imag**2)
    else:
        disp = img_np[0]

    # 颜色映射（固定色板，按 track_id 取模）
    color_palette = [
        'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta',
        'white', 'lime', 'dodgerblue', 'violet', 'gold', 'turquoise',
        'salmon', 'orchid'
    ]
    def color_for_id(tid: int):
        return color_palette[abs(int(tid)) % len(color_palette)]

    # 绘图
    target_dpi = dpi
    fig = plt.figure(figsize=(W / target_dpi, H / target_dpi), dpi=target_dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111)
    ax.imshow(disp, cmap='viridis', origin='upper')
    ax.axis('off')
    ax.set_title("Tracks & Trajectories", fontsize=10)

    # 绘制跟踪框与标签
    for i in range(tracks_np.shape[0]):
        x1, y1, x2, y2, track_cls, track_id = tracks_np[i].tolist()
        w_box = max(0.0, x2 - x1)
        h_box = max(0.0, y2 - y1)
        color = color_for_id(track_id)
        rect = Rectangle((x1, y1), w_box, h_box, linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 6),
                f"id{int(track_id)} c{int(track_cls)}",
                color=color, fontsize=8, backgroundcolor='black')

        # 轨迹：使用历史中心点连接线
        if track_histories is not None:
            hist = track_histories.get(int(track_id), [])
            cx = x1 + w_box / 2.0
            cy = y1 + h_box / 2.0
            if len(hist) >= 1:
                pts = np.array(hist + [(cx, cy)], dtype=np.float32)
            else:
                pts = np.array([(cx, cy)], dtype=np.float32)

            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.9)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color)

    # 保存到图片
    if save_path is None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "tracks_with_trajectories.png")

    # 将当前 Matplotlib 图转换为帧并写入视频
    if video_path is not None:
        # 从 canvas 抓取 RGB 像素
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 3)  # (H,W,3)
        # OpenCV 需要 BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        vw = _get_or_create_vwriter(video_path, frame_bgr.shape[1], frame_bgr.shape[0], fps=video_fps, codec=video_codec)
        vw.write(frame_bgr)

    if save_path is not None:
        fig.savefig(save_path, dpi=target_dpi)
    if show:
        plt.show()
    plt.close(fig)


def analyze_and_visualize_wm_diff(
    dataset: RadarWindowDataset,
    tolerance_pos: float = 3.0,
    tolerance_size_ratio: float = 0.10,
    reference_group: int = 1,
    topn_frames: int = 10,
    show: bool = True
):
    """
    分析多组 (group) 之间同一帧 'wm' 标注差异。
    文件命名: data{group}_{frame}.mat
      group: 1..G
      frame: 001..N
    对比内容:
      - 缺失: 参考组有 obj_id, 其他组没有
      - 位置差异: 中心距离 > tolerance_pos
      - 尺寸差异: |w_other - w_ref|/w_ref 或 |h_other - h_ref|/h_ref > tolerance_size_ratio
      - 额外: 其他组出现参考组不存在的 obj_id
    可视化:
      - 各组缺失总数条形图
      - 位置差异分布直方图
      - 尺寸差异分布直方图 (宽/高)
      - 每帧综合差异折线图
      - Top N 差异最大的帧叠加框展示
    返回:
      {
        'groups': [1,2,...],
        'frames': sorted_frame_ids,
        'per_group_missing': {g: count},
        'per_group_extra': {g: count},
        'pos_diffs': {g: [dist,...]},
        'size_w_ratio': {g: [ratio,...]},
        'size_h_ratio': {g: [ratio,...]},
        'frame_summary': [
            {
              'frame': frame_id,
              'total_ref': K,
              'per_group': {
                  g: {
                      'missing': m,
                      'extra': e,
                      'pos_bad': pb,
                      'size_bad': sb
                  }, ...
              },
              'score': score_for_ranking
            }, ...
        ],
        'top_frames': [... same as subset ...]
      }
    """
    import re
    from collections import defaultdict, Counter

    # 解析文件名并按帧聚合 wm 标注
    pattern = re.compile(r"data(\d+)_(\d+)\.mat$")
    frame_to_group_boxes = defaultdict(lambda: defaultdict(list))
    groups_set = set()
    frames_set = set()

    for file, boxes in dataset.file_to_boxes.items():
        m = pattern.match(file)
        if not m:
            continue
        g = int(m.group(1))
        f = int(m.group(2))  # 帧号整数
        groups_set.add(g)
        frames_set.add(f)
        for b in boxes:
            if b["class_name"] == "wm":
                frame_to_group_boxes[f][g].append(b)

    groups = sorted(groups_set)
    frames = sorted(frames_set)
    if reference_group not in groups:
        raise ValueError(f"参考组 {reference_group} 不在数据集中可用组 {groups}")

    # 统计结构
    per_group_missing = Counter()
    per_group_extra = Counter()
    pos_diffs = {g: [] for g in groups if g != reference_group}
    size_w_ratio = {g: [] for g in groups if g != reference_group}
    size_h_ratio = {g: [] for g in groups if g != reference_group}
    frame_summary = []

    # 主循环
    for f in frames:
        ref_boxes_list = frame_to_group_boxes[f].get(reference_group, [])
        # 参考 group 按 obj_id 建立索引
        ref_by_id = {b["obj_id"]: b for b in ref_boxes_list}
        total_ref = len(ref_by_id)
        per_group_frame_stats = {}
        score_components = []

        for g in groups:
            if g == reference_group:
                continue
            g_boxes_list = frame_to_group_boxes[f].get(g, [])
            g_by_id = {b["obj_id"]: b for b in g_boxes_list}

            # 缺失
            missing_ids = [oid for oid in ref_by_id.keys() if oid not in g_by_id]
            per_group_missing[g] += len(missing_ids)

            # 额外
            extra_ids = [oid for oid in g_by_id.keys() if oid not in ref_by_id]
            per_group_extra[g] += len(extra_ids)

            pos_bad = 0
            size_bad = 0

            # 对齐共同 obj_id
            common_ids = [oid for oid in ref_by_id.keys() if oid in g_by_id]
            for oid in common_ids:
                rb = ref_by_id[oid]
                gb = g_by_id[oid]
                # 中心
                cx_r = rb["x1"] + rb["w"] / 2.0
                cy_r = rb["y1"] + rb["h"] / 2.0
                cx_g = gb["x1"] + gb["w"] / 2.0
                cy_g = gb["y1"] + gb["h"] / 2.0
                dist = math.sqrt((cx_r - cx_g) ** 2 + (cy_r - cy_g) ** 2)
                pos_diffs[g].append(dist)
                bad_pos = dist > tolerance_pos

                # 尺寸
                w_r = rb["w"]; h_r = rb["h"]
                w_g = gb["w"]; h_g = gb["h"]
                wr = abs(w_g - w_r) / (w_r + 1e-6)
                hr = abs(h_g - h_r) / (h_r + 1e-6)
                size_w_ratio[g].append(wr)
                size_h_ratio[g].append(hr)
                bad_size = (wr > tolerance_size_ratio) or (hr > tolerance_size_ratio)

                if bad_pos:
                    pos_bad += 1
                if bad_size:
                    size_bad += 1

            per_group_frame_stats[g] = {
                "missing": len(missing_ids),
                "extra": len(extra_ids),
                "pos_bad": pos_bad,
                "size_bad": size_bad
            }
            score_components.append(len(missing_ids) + pos_bad + size_bad)

        frame_summary.append({
            "frame": f,
            "total_ref": total_ref,
            "per_group": per_group_frame_stats,
            "score": sum(score_components)
        })

    # 排名获取 top N
    frame_summary_sorted = sorted(frame_summary, key=lambda x: x["score"], reverse=True)
    top_frames = frame_summary_sorted[:topn_frames]

    # 可视化
    if show:
        import matplotlib.pyplot as plt
        # 缺失统计
        if per_group_missing:
            plt.figure(figsize=(10,4))
            gs = [g for g in groups if g != reference_group]
            miss_vals = [per_group_missing[g] for g in gs]
            extra_vals = [per_group_extra[g] for g in gs]
            x = np.arange(len(gs))
            width = 0.35
            plt.bar(x - width/2, miss_vals, width=width, label="缺失(wm)")
            plt.bar(x + width/2, extra_vals, width=width, label="额外(wm)")
            plt.xticks(x, [f"G{g}" for g in gs])
            plt.ylabel("数量")
            plt.title(f"各组相对参考组 G{reference_group} 的缺失与额外 wm 数")
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 位置差异分布
        all_pos = []
        for g, vals in pos_diffs.items():
            if vals:
                all_pos.extend(vals)
        if all_pos:
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            plt.hist(all_pos, bins=40, color='steelblue', alpha=0.8)
            plt.axvline(tolerance_pos, color='red', linestyle='--', label='tolerance_pos')
            plt.title("中心距离总体分布")
            plt.xlabel("distance")
            plt.ylabel("count")
            plt.legend()
            plt.grid(alpha=0.3)

            plt.subplot(1,2,2)
            for g, vals in pos_diffs.items():
                if vals:
                    plt.hist(vals, bins=40, alpha=0.5, label=f"G{g}")
            plt.axvline(tolerance_pos, color='red', linestyle='--', label='tolerance_pos')
            plt.title("中心距离按组分布")
            plt.xlabel("distance")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 尺寸差异分布
        all_wr = []; all_hr = []
        for g in size_w_ratio:
            all_wr.extend(size_w_ratio[g])
            all_hr.extend(size_h_ratio[g])
        if all_wr or all_hr:
            plt.figure(figsize=(12,4))
            plt.subplot(1,2,1)
            if all_wr:
                plt.hist(all_wr, bins=40, color='orange', alpha=0.75)
                plt.axvline(tolerance_size_ratio, color='red', linestyle='--', label='tolerance')
                plt.title("宽度相对误差分布")
                plt.xlabel("abs(Δw)/w_ref")
                plt.ylabel("count")
                plt.legend()
                plt.grid(alpha=0.3)
            plt.subplot(1,2,2)
            if all_hr:
                plt.hist(all_hr, bins=40, color='green', alpha=0.75)
                plt.axvline(tolerance_size_ratio, color='red', linestyle='--', label='tolerance')
                plt.title("高度相对误差分布")
                plt.xlabel("abs(Δh)/h_ref")
                plt.ylabel("count")
                plt.legend()
                plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # 每帧差异评分折线
        if frame_summary:
            plt.figure(figsize=(12,4))
            xs = [fs["frame"] for fs in frame_summary]
            ys = [fs["score"] for fs in frame_summary]
            plt.plot(xs, ys, marker='o', linewidth=1.2, color='purple')
            plt.title("每帧综合差异评分 (缺失+位置+尺寸)")
            plt.xlabel("frame")
            plt.ylabel("score")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Top N 帧可视化
        if top_frames:
            # 颜色
            color_map = {
                reference_group: 'lime',
                # 其他组
            }
            other_colors = ['red', 'cyan', 'yellow', 'magenta', 'orange', 'white']
            oc_idx = 0
            for g in groups:
                if g == reference_group:
                    continue
                color_map[g] = other_colors[oc_idx % len(other_colors)]
                oc_idx += 1

            for tf in top_frames:
                f_id = tf["frame"]
                ref_file = f"data{reference_group}_{f_id:03d}.mat"
                try:
                    full_ref = dataset._load_full_matrix(ref_file)
                except Exception:
                    continue
                # dB 显示
                eps = 1e-12
                amp = np.abs(full_ref) if np.iscomplexobj(full_ref) else np.abs(full_ref.astype(np.float32))
                db_full = 20 * np.log10(amp + eps)
                v1, v2 = np.percentile(db_full, [1, 99])
                disp = np.clip((db_full - v1) / (v2 - v1 + 1e-6), 0, 1)

                plt.figure(figsize=(7,6))
                plt.imshow(disp, cmap='viridis', origin='upper')
                plt.title(f"Top差异帧 frame={f_id} score={tf['score']}")
                plt.axis('off')

                # 画参考组
                ref_boxes_list = frame_to_group_boxes[f_id].get(reference_group, [])
                for b in ref_boxes_list:
                    x1 = b["x1"]; y1 = b["y1"]; w = b["w"]; h = b["h"]
                    rect = Rectangle((x1, y1), w, h, edgecolor=color_map[reference_group],
                                     facecolor='none', linewidth=1.2)
                    plt.gca().add_patch(rect)
                    plt.text(x1, max(0, y1 - 5), f"ref id{b['obj_id']}", color='white',
                             fontsize=7, backgroundcolor='black')

                # 其他组
                for g in groups:
                    if g == reference_group:
                        continue
                    g_boxes_list = frame_to_group_boxes[f_id].get(g, [])
                    for b in g_boxes_list:
                        tag = f"G{g} id{b['obj_id']}"
                        # 判断是否匹配参考且超差
                        refb = next((rb for rb in ref_boxes_list if rb["obj_id"] == b["obj_id"]), None)
                        bad_flag = ""
                        if refb is not None:
                            cx_r = refb["x1"] + refb["w"]/2
                            cy_r = refb["y1"] + refb["h"]/2
                            cx_g = b["x1"] + b["w"]/2
                            cy_g = b["y1"] + b["h"]/2
                            dist = math.sqrt((cx_r - cx_g)**2 + (cy_r - cy_g)**2)
                            wr = abs(b["w"] - refb["w"]) / (refb["w"] + 1e-6)
                            hr = abs(b["h"] - refb["h"]) / (refb["h"] + 1e-6)
                            if dist > tolerance_pos or wr > tolerance_size_ratio or hr > tolerance_size_ratio:
                                bad_flag = " *"
                        x1 = b["x1"]; y1 = b["y1"]; w = b["w"]; h = b["h"]
                        rect = Rectangle((x1, y1), w, h, edgecolor=color_map[g],
                                         facecolor='none', linewidth=1.0, linestyle='--')
                        plt.gca().add_patch(rect)
                        plt.text(x1, y1, tag + bad_flag, color=color_map[g],
                                 fontsize=7, backgroundcolor='black')
                # 图例
                legend_patches = [Rectangle((0,0),1,1, edgecolor=color_map[g], facecolor='none', label=f"G{g}") for g in groups]
                plt.legend(handles=legend_patches, loc='lower right', fontsize=8)
                plt.tight_layout()
                plt.show()

    return {
        "groups": groups,
        "frames": frames,
        "per_group_missing": dict(per_group_missing),
        "per_group_extra": dict(per_group_extra),
        "pos_diffs": pos_diffs,
        "size_w_ratio": size_w_ratio,
        "size_h_ratio": size_h_ratio,
        "frame_summary": frame_summary,
        "top_frames": top_frames,
        "tolerance_pos": tolerance_pos,
        "tolerance_size_ratio": tolerance_size_ratio,
        "reference_group": reference_group
    }

