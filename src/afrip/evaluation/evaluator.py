"""目标检测评估器：mAP / PD / PFA 指标计算 + 可视化。"""
from __future__ import annotations

import os

import numpy as np
import torch

from afrip.evaluation.map_metrics import MAPCalculator
from afrip.evaluation.visualize import visualize_full_predictions, plot_roc_curve


class Evaluator:
    """在测试集上评估检测模型，计算 mAP、PD、PFA，并可选地保存最优权重。

    Args:
        cfg: 合并后的实验配置字典。期望字段：
             - ``strategy.eval.iou_thresh``  检测匹配 IoU 阈值，默认 0.5
             - ``strategy.eval.save_folder`` 权重保存目录
             - ``strategy.vis_pred_full``    是否可视化全帧预测
             - ``detector.postprocessor_cfg.conf_thresh`` 推理置信度阈值
    """

    def __init__(self, cfg: dict):
        self.cfg          = cfg
        eval_cfg          = cfg.get("strategy", {}).get("eval", {})
        post_cfg          = cfg.get("detector", {}).get("postprocessor_cfg", {})
        self.iou_thresh   = eval_cfg.get("iou_thresh", 0.5)
        self.save_folder  = eval_cfg.get("save_folder", "outputs/weights")
        self.conf_thresh  = post_cfg.get("conf_thresh", 0.01)
        self.vis_full     = cfg.get("strategy", {}).get("vis_pred_full", False)

    @torch.no_grad()
    def evaluate_with_details(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        epoch: int = 0,
        best_map: float = float("-inf"),
        save_best: bool = False,
    ) -> dict:
        return self._evaluate_impl(
            model=model,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            best_map=best_map,
            save_best=save_best,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        epoch: int,
        best_map: float,
    ) -> float:
        """在 test_loader 上推理，计算指标，可选保存最优权重。

        Args:
            model:       检测模型。
            test_loader: 验证集 DataLoader（batch_size=1，full_frame）。
            device:      推理设备。
            epoch:       当前 epoch（0-indexed）。
            best_map:    当前最优 mAP，用于判断是否保存权重。

        Returns:
            更新后的 best_map。
        """
        details = self._evaluate_impl(
            model=model,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            best_map=best_map,
            save_best=True,
        )
        return float(details["best_map"])

    @torch.no_grad()
    def _evaluate_impl(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        epoch: int,
        best_map: float,
        save_best: bool,
    ) -> dict:
        prev_training_mode = model.training
        model.eval()

        # ── 统计容器 ─────────────────────────────────────────
        ground_truth_data:    dict[str, list[dict]]  = {}
        dr_data:              dict[str, list[dict]]  = {}
        gt_counter_per_class: dict[str, int]         = {}
        images_per_class:     dict[str, set[str]]    = {}
        prediction_records:   list[dict]             = []
        total_non_target_cells = 0
        class_names_seen: set[str] = set()
        default_class    = "target"

        for batch in test_loader:
            images = batch["images"].to(device, non_blocking=True).float()
            H, W   = images.shape[-2], images.shape[-1]
            metas  = batch.get("batch_meta", None)

            # ── GT 组装 ───────────────────────────────────────
            gt_xyxy = batch["targets"][0]["boxes"].to(dtype=torch.float32).clone()
            if gt_xyxy.numel() > 0:
                mask = torch.zeros((H, W), dtype=torch.bool)
                for b in gt_xyxy:
                    x1 = int(b[0].clamp(0, W).item())
                    y1 = int(b[1].clamp(0, H).item())
                    x2 = int(b[2].clamp(0, W).item())
                    y2 = int(b[3].clamp(0, H).item())
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = True
                total_non_target_cells += int(H * W - mask.sum().item())
            else:
                gt_xyxy = torch.zeros((0, 4), dtype=torch.float32)
                total_non_target_cells += H * W

            file_id = (metas[0]["file"]
                       if metas is not None
                       else f"sample_{len(ground_truth_data)}")

            gt_list: list[dict] = []
            for j in range(gt_xyxy.shape[0]):
                x1, y1, x2, y2 = gt_xyxy[j].tolist()
                cls_name = default_class
                class_names_seen.add(cls_name)
                gt_list.append({
                    "class_name": cls_name,
                    "bbox":       f"{x1} {y1} {x2} {y2}",
                    "used":       False,
                    "difficult":  False,
                })
                gt_counter_per_class[cls_name] = gt_counter_per_class.get(cls_name, 0) + 1
                images_per_class.setdefault(cls_name, set()).add(file_id)
            ground_truth_data[file_id] = gt_list

            # ── 推理 ─────────────────────────────────────────
            preds = model(images)
            pred_bboxes = preds["boxes"]
            pred_scores = preds["scores"]

            # 过滤非法框
            if pred_bboxes.numel() > 0:
                valid  = torch.isfinite(pred_bboxes).all(dim=1)
                proper = (pred_bboxes[:, 2] > pred_bboxes[:, 0]) & (pred_bboxes[:, 3] > pred_bboxes[:, 1])
                keep   = valid & proper
                pred_bboxes = pred_bboxes[keep]
                pred_scores = pred_scores[keep]

            # 按分数降序排列
            if pred_scores.numel() > 0:
                order       = torch.argsort(pred_scores, descending=True)
                pred_bboxes = pred_bboxes[order]
                pred_scores = pred_scores[order]

            # 填入 dr_data
            for j in range(pred_bboxes.shape[0]):
                x1, y1, x2, y2 = pred_bboxes[j].tolist()
                score    = float(pred_scores[j].item())
                cls_name = default_class
                class_names_seen.add(cls_name)
                dr_data.setdefault(cls_name, []).append({
                    "confidence": score,
                    "file_id":    file_id,
                    "bbox":       f"{x1} {y1} {x2} {y2}",
                })

            # 缓存可视化数据
            if metas is not None:
                y0, x0 = metas[0]["global_origin"]
                prediction_records.append({
                    "file":        file_id,
                    "origin":      (y0, x0),
                    "pred_boxes":  pred_bboxes.cpu(),
                    "pred_scores": pred_scores.cpu(),
                    "gt_boxes":    gt_xyxy.cpu(),
                })

        # ── 确保各类预测按置信度降序 ──────────────────────────
        for cls in dr_data:
            dr_data[cls].sort(key=lambda x: -x["confidence"])

        counter_images_per_class = {
            cls: len(images_per_class.get(cls, set()))
            for cls in class_names_seen
        }

        # ── 计算 mAP ─────────────────────────────────────────
        map_calc = MAPCalculator(min_overlap=self.iou_thresh)
        map_res  = map_calc.evaluate(
            ground_truth_data=ground_truth_data,
            dr_data=dr_data,
            gt_counter_per_class=gt_counter_per_class,
            counter_images_per_class=counter_images_per_class,
        )
        ap       = float(map_res["mAP"])
        count_tp = map_res["count_tp"]
        total_gt = sum(gt_counter_per_class.values())
        total_tp = sum(count_tp.values())
        total_fp = sum(
            len(dr_data.get(cls, [])) - count_tp.get(cls, 0)
            for cls in class_names_seen
        )
        pd  = float(total_tp / max(total_gt, 1))
        pfa = float(total_fp / max(total_non_target_cells, 1))

        # ── 保存最优权重 ──────────────────────────────────────
        if save_best and ap > best_map:
            os.makedirs(self.save_folder, exist_ok=True)
            save_path = os.path.join(
                self.save_folder,
                f"best_epoch_{epoch + 1}_map_{ap:.4f}.pth",
            )
            torch.save(model.state_dict(), save_path)
            best_map = ap
            print(f"[Eval] New best mAP={ap:.4f}, PD={pd:.4f}, PFA={pfa:.6f} → saved: {save_path}")
        else:
            print(f"[Eval] mAP={ap:.4f}, PD={pd:.4f}, PFA={pfa:.6f}")

        # ── 恢复模型训练状态 ──────────────────────────────────
        model.train(prev_training_mode)

        # ── 可视化 ────────────────────────────────────────────
        if self.vis_full and prediction_records:
            try:
                dataset = test_loader.dataset
                print("[Eval] 可视化整幅检测结果")
                visualize_full_predictions(
                    dataset=dataset,
                    records=prediction_records,
                    conf_thresh=self.conf_thresh,
                    iou_thresh=self.iou_thresh,
                    max_files=10,
                    epoch=epoch + 1,
                )
                if default_class in map_res.get("pr_curves", {}):
                    plot_roc_curve(
                        pr_curve=map_res["pr_curves"][default_class],
                        total_gt=total_gt,
                        total_non_target_cells=total_non_target_cells,
                        epoch=epoch,
                    )
            except Exception as e:
                print(f"[Eval] 可视化失败: {e}")

        return {
            "best_map": best_map,
            "mAP": ap,
            "pd": pd,
            "pfa": pfa,
            "total_gt": total_gt,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_non_target_cells": total_non_target_cells,
            "class_names_seen": sorted(class_names_seen),
            "default_class": default_class,
            "map_res": map_res,
        }
