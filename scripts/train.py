"""AFRIP 训练入口 — 基于 YAML 配置，对应旧版 train_yoloRT.py。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

import numpy as np
import torch

from afrip.utils import load_config
from afrip.engine import Trainer
from afrip.evaluation import Evaluator


# ─────────────────────────────── CLI ────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFRIP Radar Detection Training")
    parser.add_argument("--config", default='./configs/experiments/detection/radardet_yolortv2_sort.yaml', help="实验配置 YAML 路径")
    # 运行时覆盖（对应旧版 ExperimentConfig 字段）
    parser.add_argument("--device",      type=str,   default=None, help="覆盖设备 (cpu/cuda/cuda:0)")
    parser.add_argument("--batch_size",  type=int,   default=None, help="覆盖 batch size")
    parser.add_argument("--epochs",      type=int,   default=None, help="覆盖最大 epoch 数")
    parser.add_argument("--fp16",        action="store_true",       help="启用混合精度训练")
    parser.add_argument("--resume",      type=str,   default=None, help="checkpoint 路径，用于恢复训练")
    parser.add_argument("--save_folder", type=str,   default=None, help="覆盖权重保存目录")
    parser.add_argument("--conf_thresh", type=float, default=None, help="覆盖推理置信度阈值")
    parser.add_argument("--nms_thresh",  type=float, default=None, help="覆盖 NMS IoU 阈值")
    parser.add_argument("--iou_thresh",  type=float, default=None, help="覆盖评估 IoU 阈值")
    return parser.parse_args()


# ─────────────────────────── 覆盖工具 ───────────────────────────────

def _nested_set(d: dict, keys: list[str], value: Any) -> None:
    """在嵌套字典中按路径设置值，中间层不存在则自动创建。"""
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """将命令行参数覆盖到配置字典（就地修改）。"""
    if args.device:
        _nested_set(cfg, ["runtime", "device"], args.device)
    if args.batch_size:
        _nested_set(cfg, ["dataloader", "batch_size"], args.batch_size)
    if args.epochs:
        _nested_set(cfg, ["strategy", "train", "max_epoch"], args.epochs)
    if args.fp16:
        _nested_set(cfg, ["strategy", "train", "fp16"], True)
    if args.resume:
        _nested_set(cfg, ["strategy", "train", "resume"], args.resume)
    if args.save_folder:
        _nested_set(cfg, ["strategy", "eval", "save_folder"], args.save_folder)
    if args.conf_thresh is not None:
        _nested_set(cfg, ["detector", "postprocessor_cfg", "conf_thresh"], args.conf_thresh)
    if args.nms_thresh is not None:
        _nested_set(cfg, ["detector", "postprocessor_cfg", "nms_thresh"], args.nms_thresh)
    if args.iou_thresh is not None:
        _nested_set(cfg, ["strategy", "eval", "iou_thresh"], args.iou_thresh)


# ─────────────────────────── 随机种子 ───────────────────────────────

def apply_seed(cfg: dict) -> None:
    """若配置了 deterministic=true，则固定所有随机源。"""
    if not cfg.get("deterministic", False):
        return
    seed = cfg.get("runtime", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] deterministic=True, seed={seed}")


# ─────────────────────────── 配置摘要 ───────────────────────────────

def print_summary(cfg: dict) -> None:
    """对应旧版 ExperimentConfig.summary()，打印关键配置节。"""
    print("=" * 45)
    print("  AFRIP Experiment Configuration")
    print("=" * 45)
    sections = ("experiment", "runtime", "dataset", "detector", "loss", "strategy")
    for sec in sections:
        val = cfg.get(sec)
        if val is None:
            continue
        print(f"\n[{sec}]")
        if isinstance(val, dict):
            # 只打印一层，嵌套 dict 以缩进形式展示
            for k, v in val.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for kk, vv in v.items():
                        print(f"    {kk}: {vv}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {val}")
    print("=" * 45)


# ─────────────────────────── 主函数 ─────────────────────────────────

def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    apply_seed(cfg)
    print_summary(cfg)

    trainer   = Trainer(cfg)
    evaluator = Evaluator(cfg)
    best_map  = trainer.run(evaluator=evaluator)

    print(f"\nTraining finished. Best mAP: {best_map:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

