"""mAP / LAMR 计算工具，遵循 VOC 评估协议。"""
from __future__ import annotations

import copy
import math

import numpy as np

MINOVERLAP = 0.5


def voc_ap(
    rec: list[float],
    prec: list[float],
) -> tuple[float, list[float], list[float]]:
    """计算 VOC 11 点（11-point interpolation）AP。

    Args:
        rec:  召回率列表（未填充端点）。
        prec: 精确率列表（未填充端点）。

    Returns:
        (ap, smoothed_rec, smoothed_prec)
    """
    rec  = [0.0] + rec  + [1.0]
    prec = [0.0] + prec + [0.0]
    # 精确率包络线
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    i_list = [i for i in range(1, len(rec)) if rec[i] != rec[i - 1]]
    ap     = sum((rec[i] - rec[i - 1]) * prec[i] for i in i_list)
    return ap, rec, prec


def log_average_miss_rate(
    prec: np.ndarray,
    rec: np.ndarray,
    num_images: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """计算 log 平均漏检率（LAMR）。

    Args:
        prec:       精确率数组。
        rec:        召回率数组。
        num_images: 图像总数（用于 FPPI 归一化）。

    Returns:
        (lamr, mr, fppi)
    """
    if prec.size == 0:
        return 0.0, np.array([1.0]), np.array([0.0])

    fppi = 1 - prec
    mr   = 1 - rec
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp   = np.insert(mr,   0,  1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j      = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


class MAPCalculator:
    """基于 VOC 协议的 mAP 计算器。

    Args:
        min_overlap:  默认 IoU 阈值，默认 0.5。
        specific_iou: 各类别单独覆盖的 IoU 阈值 {class_name: iou}，可选。
    """

    def __init__(
        self,
        min_overlap: float = MINOVERLAP,
        specific_iou: dict[str, float] | None = None,
    ):
        self.min_overlap  = min_overlap
        self.specific_iou = specific_iou or {}

    def evaluate(
        self,
        ground_truth_data: dict[str, list[dict]],
        dr_data: dict[str, list[dict]],
        gt_counter_per_class: dict[str, int],
        counter_images_per_class: dict[str, int],
    ) -> dict:
        """执行 mAP 评估。

        Args:
            ground_truth_data:        {file_id: [{class_name, bbox, used, difficult}]}
            dr_data:                  {class_name: [{confidence, file_id, bbox}]}（已按置信度降序排列）
            gt_counter_per_class:     {class_name: gt_count}
            counter_images_per_class: {class_name: n_images}

        Returns:
            dict 包含 ap_dict、lamr_dict、count_tp、mAP、classes、pr_curves。
        """
        ap_dict    = {}
        lamr_dict  = {}
        pr_curves  = {}
        count_tp   = {cls: 0 for cls in gt_counter_per_class}
        sum_ap     = 0.0
        classes    = sorted(gt_counter_per_class.keys())

        # 深拷贝 GT，保证 used 标记独立
        gt_used = {fid: copy.deepcopy(items) for fid, items in ground_truth_data.items()}

        for cls in classes:
            detections = dr_data.get(cls, [])
            tp = [0] * len(detections)
            fp = [0] * len(detections)
            min_ov = self.specific_iou.get(cls, self.min_overlap)

            for idx, det in enumerate(detections):
                file_id = det["file_id"]
                bb      = [float(x) for x in det["bbox"].split()]
                ovmax, match = -1.0, None

                for obj in gt_used.get(file_id, []):
                    if obj["class_name"] != cls:
                        continue
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi   = [
                        max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                        min(bb[2], bbgt[2]), min(bb[3], bbgt[3]),
                    ]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = ((bb[2]   - bb[0]   + 1) * (bb[3]   - bb[1]   + 1)
                             + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                             - iw * ih)
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax, match = ov, obj

                if ovmax >= min_ov and match and not match.get("difficult", False):
                    if not match.get("used", False):
                        tp[idx]    = 1
                        match["used"] = True
                        count_tp[cls] += 1
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            # 累积 TP / FP
            for i in range(1, len(fp)):
                fp[i] += fp[i - 1]
                tp[i] += tp[i - 1]

            gt_n = gt_counter_per_class[cls]
            rec  = [tp[i] / gt_n for i in range(len(tp))]
            prec = [
                tp[i] / (fp[i] + tp[i]) if (fp[i] + tp[i]) > 0 else 0.0
                for i in range(len(tp))
            ]

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_ap  += ap
            ap_dict[cls]  = ap
            pr_curves[cls] = {"rec": rec, "prec": prec, "mrec": mrec, "mprec": mprec}

            n_img = counter_images_per_class.get(cls, 0)
            lamr, _, _ = log_average_miss_rate(np.array(prec), np.array(rec), n_img)
            lamr_dict[cls] = lamr

        mAP = sum_ap / len(classes) if classes else 0.0
        return {
            "ap_dict":   ap_dict,
            "lamr_dict": lamr_dict,
            "count_tp":  count_tp,
            "mAP":       mAP,
            "classes":   classes,
            "pr_curves": pr_curves,
        }
