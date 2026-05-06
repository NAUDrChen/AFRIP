"""AFRIP data1 前 50 帧 ROC 评估入口。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from afrip.datasets import build_dataset, build_transform_pipeline
from afrip.evaluation import Evaluator
from afrip.evaluation.visualize import plot_roc_curve
from afrip.models import build_detector
from afrip.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFRIP ROC evaluation on the first 50 frames of data1")
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            "./configs/experiments/detection/radardet_rdcnn_sort.yaml",
            "./configs/experiments/detection/radardet_yolortv2_sort.yaml",
        ],
        help="实验配置 YAML 路径，可传多个",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=[
            r".\outputs\radardet_rdcnn_sort\weights\best_epoch_6_map_0.5174-PD=0.7161, PFA=0.000018.pth",
            r".\outputs\radardet_yolortv2_sort\weights\best_epoch_10_map_0.6752.pth",
        ],
        help="checkpoint 路径；数量需与 config 对齐，或省略后自动解析",
    )
    parser.add_argument("--sequence_name", type=str, default="data1", help="要筛选的数据序列名")
    parser.add_argument("--max_frames", type=int, default=50, help="最多评估多少帧")
    parser.add_argument("--device", type=str, default=None, help="覆盖推理设备")
    parser.add_argument("--conf_thresh", type=float, default=None, help="覆盖推理置信度阈值")
    parser.add_argument("--nms_thresh", type=float, default=None, help="覆盖 NMS IoU 阈值")
    parser.add_argument("--iou_thresh", type=float, default=None, help="覆盖评估 IoU 阈值")
    parser.add_argument("--output_dir", type=str, default="outputs/roc_eval_data1_50", help="ROC 图与汇总输出目录")
    parser.add_argument("--title", type=str, default="ROC Comparison on data1 First 50 Frames", help="多实验对比图标题")
    parser.add_argument("--vis_pred_full", action="store_true", help="同时输出整幅预测可视化")
    return parser.parse_args()


def _nested_set(d: dict, keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.device:
        _nested_set(cfg, ["runtime", "device"], args.device)
    if args.conf_thresh is not None:
        _nested_set(cfg, ["detector", "conf_thresh"], args.conf_thresh)
        post_cfg = cfg.get("detector", {}).get("postprocessor_cfg")
        if isinstance(post_cfg, dict):
            post_cfg["conf_thresh"] = args.conf_thresh
    if args.nms_thresh is not None:
        _nested_set(cfg, ["detector", "nms_thresh"], args.nms_thresh)
        post_cfg = cfg.get("detector", {}).get("postprocessor_cfg")
        if isinstance(post_cfg, dict):
            post_cfg["nms_thresh"] = args.nms_thresh
    if args.iou_thresh is not None:
        _nested_set(cfg, ["strategy", "eval", "iou_thresh"], args.iou_thresh)
    if args.vis_pred_full:
        _nested_set(cfg, ["strategy", "vis_pred_full"], True)


def _matches_sequence_name(file_name: str, sequence_name: str) -> bool:
    token = sequence_name.strip().lower()
    if not token:
        raise ValueError("--sequence_name 不能为空")

    normalized = file_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    candidates = {part.lower() for part in path.parts}
    candidates.add(path.name.lower())
    candidates.add(path.stem.lower())

    for candidate in candidates:
        if candidate == token:
            return True
        if candidate.startswith(f"{token}_") or candidate.startswith(f"{token}-"):
            return True
    return False


def filter_dataset_to_sequence_frames(dataset: Any, sequence_name: str, max_frames: int) -> list[str]:
    if max_frames <= 0:
        raise ValueError("--max_frames 必须大于 0")

    matched_index = [entry for entry in dataset.index if _matches_sequence_name(entry[0], sequence_name)]
    if not matched_index:
        raise RuntimeError(f"测试集中未找到匹配序列 {sequence_name!r} 的样本")
    if len(matched_index) < max_frames:
        raise RuntimeError(
            f"序列 {sequence_name!r} 仅匹配到 {len(matched_index)} 帧，少于要求的 {max_frames} 帧"
        )

    limited_index = matched_index[:max_frames]
    selected_files = [file_name for file_name, _, _ in limited_index]
    dataset.index = limited_index
    dataset.files = list(dict.fromkeys(selected_files))
    return selected_files


def build_val_loader(cfg: dict, args: argparse.Namespace) -> tuple[DataLoader, list[str]]:
    ds_cfg = dict(cfg["dataset"])
    val_pipeline = build_transform_pipeline(cfg.get("val_transforms"))
    val_ds = build_dataset(
        {**ds_cfg, "subset": "test", "full_frame": True},
        transforms=val_pipeline,
    )
    selected_files = filter_dataset_to_sequence_frames(val_ds, args.sequence_name, args.max_frames)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("strategy", {}).get("train", {}).get("num_workers_test", 0),
        collate_fn=val_ds.collate_fn,
        pin_memory=True,
    )
    return val_loader, selected_files


def load_checkpoint_to_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    model.load_state_dict(state_dict, strict=True)


def resolve_checkpoint(cfg: dict, explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"checkpoint 不存在: {explicit_path}")
        return explicit_path

    resume_path = cfg.get("strategy", {}).get("eval", {}).get("resume")
    if resume_path and os.path.exists(resume_path):
        return resume_path

    save_folder = cfg.get("strategy", {}).get("eval", {}).get("save_folder")
    if save_folder and os.path.isdir(save_folder):
        candidates = sorted(
            Path(save_folder).glob("*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    raise FileNotFoundError("无法从 --checkpoint、strategy.eval.resume 或 save_folder 中解析 checkpoint")


def evaluate_single_experiment(config_path: str, checkpoint_path: str | None, args: argparse.Namespace) -> dict:
    cfg = load_config(config_path)
    apply_overrides(cfg, args)

    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    detector_cfg = {**cfg["detector"], "trainable": False, "deploy": False}
    model = build_detector(detector_cfg).to(device)

    resolved_checkpoint = resolve_checkpoint(cfg, checkpoint_path)
    load_checkpoint_to_model(model, resolved_checkpoint, device)

    val_loader, selected_files = build_val_loader(cfg, args)
    evaluator = Evaluator(cfg)
    details = evaluator.evaluate_with_details(
        model=model,
        test_loader=val_loader,
        device=device,
        epoch=0,
        best_map=float("-inf"),
        save_best=False,
    )

    exp_name = cfg.get("experiment", {}).get("name") or Path(config_path).stem
    default_class = details["default_class"]
    pr_curve = details["map_res"].get("pr_curves", {}).get(default_class)
    if pr_curve is None:
        raise RuntimeError(f"未找到类别 {default_class} 的 PR 曲线，无法绘制 ROC")

    return {
        "experiment_name": exp_name,
        "config_path": config_path,
        "checkpoint_path": resolved_checkpoint,
        "sequence_name": args.sequence_name,
        "frame_count": len(selected_files),
        "selected_files": selected_files,
        "details": details,
        "pr_curve": pr_curve,
    }


def save_summary(results: list[dict], output_dir: str) -> str:
    summary = []
    for result in results:
        details = result["details"]
        summary.append(
            {
                "experiment_name": result["experiment_name"],
                "config_path": result["config_path"],
                "checkpoint_path": result["checkpoint_path"],
                "sequence_name": result["sequence_name"],
                "frame_count": result["frame_count"],
                "selected_files": result["selected_files"],
                "mAP": details["mAP"],
                "pd": details["pd"],
                "pfa": details["pfa"],
                "total_gt": details["total_gt"],
                "total_tp": details["total_tp"],
                "total_fp": details["total_fp"],
                "total_non_target_cells": details["total_non_target_cells"],
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "roc_summary.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return out_path


def main() -> int:
    args = parse_args()
    checkpoints = args.checkpoint or []
    if checkpoints and len(checkpoints) != len(args.config):
        raise ValueError("--checkpoint 数量必须与 --config 数量一致")

    results = []
    for index, config_path in enumerate(args.config):
        checkpoint_path = checkpoints[index] if index < len(checkpoints) else None
        result = evaluate_single_experiment(config_path, checkpoint_path, args)
        results.append(result)

    os.makedirs(args.output_dir, exist_ok=True)

    for result in results:
        exp_dir = os.path.join(args.output_dir, result["experiment_name"])
        os.makedirs(exp_dir, exist_ok=True)
        roc_path = os.path.join(exp_dir, f"{result['experiment_name']}_roc.png")
        roc_meta = plot_roc_curve(
            pr_curve=result["pr_curve"],
            total_gt=result["details"]["total_gt"],
            total_non_target_cells=result["details"]["total_non_target_cells"],
            epoch=0,
            save_dir=exp_dir,
            label=result["experiment_name"],
            save_path=roc_path,
        )
        print(
            f"[ROC] {result['experiment_name']}: frames={result['frame_count']}, "
            f"sequence={result['sequence_name']}, mAP={result['details']['mAP']:.4f}, "
            f"PD={result['details']['pd']:.4f}, PFA={result['details']['pfa']:.6f}, "
            f"curve={roc_meta['out_path']}"
        )

    if len(results) > 1:
        fig, ax = plt.subplots()
        for result in results:
            plot_roc_curve(
                pr_curve=result["pr_curve"],
                total_gt=result["details"]["total_gt"],
                total_non_target_cells=result["details"]["total_non_target_cells"],
                epoch=0,
                label=result["experiment_name"],
                ax=ax,
            )
        ax.set_title(args.title)
        fig.tight_layout()
        compare_path = os.path.join(args.output_dir, "roc_comparison.png")
        fig.savefig(compare_path, dpi=150)
        plt.close(fig)
        print(f"[ROC] comparison saved: {compare_path}")

    summary_path = save_summary(results, args.output_dir)
    print(f"[ROC] summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())