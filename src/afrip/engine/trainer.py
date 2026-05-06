"""AFRIP 训练引擎：从合并后的 YAML 配置字典构建训练流水线。"""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from afrip.core import BaseDataset, BaseDetector
from afrip.datasets import build_dataset, build_transform_pipeline
from afrip.models import DetectionBatch, assemble_detection_components
from afrip.strategies import build_optimizer, build_scheduler

# 顶层全局：供 DataLoader worker 初始化函数访问
_WORKER_BASE_SEED: int | None = None


def _dataloader_worker_init_fn(worker_id: int) -> None:
    base = _WORKER_BASE_SEED or 0
    seed = int(base) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """AFRIP 训练器。

    Args:
        cfg: 由 ``load_config`` 合并后的配置字典，期望包含以下顶层键：

            - ``runtime``          设备、种子、最大 epoch、日志频率
            - ``detector``         探测器配置
            - ``loss``             损失函数配置
            - ``dataset``          数据集配置
            - ``dataloader``       DataLoader 参数（batch_size、num_workers、shuffle）
            - ``train_transforms`` 训练增强管线（列表）
            - ``val_transforms``   验证增强管线（列表，可选）
            - ``strategy``         含 optimizer / scheduler / train / eval 子字典
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

        rt        = cfg.get("runtime", {})
        strat     = cfg.get("strategy", {})
        train_cfg = strat.get("train", {})
        eval_cfg  = strat.get("eval", {})

        # ── 设备与随机种子 ────────────────────────────────────
        self.device    = torch.device(rt.get("device", "cpu"))
        self._seed     = rt.get("seed", 42)
        self._det_seed = cfg.get("deterministic", False)
        if self._det_seed:
            _set_seed(self._seed)

        # ── 训练超参 ─────────────────────────────────────────
        self.max_epoch     = rt.get("max_epochs", train_cfg.get("max_epoch", 50))
        self.fp16          = train_cfg.get("fp16", False)
        self.warmup_epoch  = train_cfg.get("warmup_epoch", 3)
        self.eval_interval = train_cfg.get("eval_interval", 5)
        self.clip_grad     = train_cfg.get("clip_grad", 0.0)
        self.log_interval  = rt.get("log_interval", 10)
        self._last_opt_step = 0

        # ── 混合精度 scaler ───────────────────────────────────
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # ── 模型装配 ─────────────────────────────────────────
        self.model, self.criterion = assemble_detection_components(
            cfg,
            trainable=True,
        )
        self.model = self.model.to(self.device)
        if isinstance(self.model, BaseDetector):
            self.model.set_training_behavior(True)

        # ── 优化器 ───────────────────────────────────────────
        optim_cfg = strat.get("optimizer", {})
        resume    = eval_cfg.get("resume", None)
        self.optimizer, self.start_epoch = build_optimizer(
            optim_cfg, self.model, resume
        )

        # ── 学习率调度器 ──────────────────────────────────────
        sched_cfg = strat.get("scheduler", {})
        self.lr_scheduler, self.lf = build_scheduler(
            sched_cfg, self.optimizer, self.max_epoch
        )
        self.lr_scheduler.last_epoch = self.start_epoch - 1
        self._sched_cfg = sched_cfg
        self._optim_cfg = optim_cfg

        # ── DataLoaders ───────────────────────────────────────
        self.train_loader, self.val_loader = self._build_dataloaders()

    # ─────────────────────────── 数据集构建 ───────────────────────────

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        dl_cfg    = self.cfg.get("dataloader", {})
        train_cfg = self.cfg.get("strategy", {}).get("train", {})
        ds_cfg    = self.cfg["dataset"]

        # 训练集：使用 train_transforms，subset=train
        train_pipeline = build_transform_pipeline(
            self.cfg.get("train_transforms")
        )
        train_ds = build_dataset(
            {**ds_cfg, "subset": "train"},
            transforms=train_pipeline,
        )

        # 验证集：使用 val_transforms（默认仅归一化），subset=test，全帧读取
        val_pipeline = build_transform_pipeline(
            self.cfg.get("val_transforms")
        )
        val_ds = build_dataset(
            {**ds_cfg, "subset": "test", "full_frame": True},
            transforms=val_pipeline,
        )

        train_collate_fn = self._resolve_collate_fn(train_ds)
        val_collate_fn = self._resolve_collate_fn(val_ds)

        # 可重现性
        global _WORKER_BASE_SEED
        g               = None
        worker_init_fn  = None
        if self._det_seed:
            g = torch.Generator()
            g.manual_seed(self._seed)
            _WORKER_BASE_SEED = self._seed
            worker_init_fn    = _dataloader_worker_init_fn

        num_workers_train = train_cfg.get(
            "num_workers_train", dl_cfg.get("num_workers", 0)
        )
        persistent = train_cfg.get("persistent_workers", False)
        batch_size  = dl_cfg.get("batch_size", 4)
        shuffle     = dl_cfg.get("shuffle", True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers_train,
            collate_fn=train_collate_fn,
            pin_memory=True,
            persistent_workers=persistent and num_workers_train > 0,
            generator=g,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=train_cfg.get("num_workers_test", 0),
            collate_fn=val_collate_fn,
            pin_memory=True,
        )
        return train_loader, val_loader

    # ─────────────────────────── 单 epoch 训练 ────────────────────────

    def train_one_epoch(self, epoch: int) -> None:
        """执行一个 epoch 的训练，含 warmup 学习率线性插值。"""
        self.model.train()
        if isinstance(self.model, BaseDetector):
            self.model.set_training_behavior(True)

        batch_size  = self.cfg.get("dataloader", {}).get("batch_size", 4)
        epoch_size  = len(self.train_loader)
        nw          = epoch_size * self.warmup_epoch
        last_opt    = self._last_opt_step
        warmup_bias = self._sched_cfg.get("warmup_bias_lr",  0.1)
        warmup_mom  = self._sched_cfg.get("warmup_momentum", 0.8)
        momentum    = self._optim_cfg.get("momentum",        0.937)

        t0 = time.time()
        for iter_i, batch in enumerate(self.train_loader):
            batch_contract = DetectionBatch.from_mapping(batch)
            images = batch_contract.images.to(self.device, non_blocking=True).float()
            targets = batch_contract.targets
            ni      = iter_i + epoch * epoch_size

            # 梯度累积步数（模拟 effective batch_size=64）
            accumulate = max(1, round(64 / batch_size))
            if ni <= nw:
                xi         = [0, nw]
                accumulate = max(
                    1, int(np.interp(ni, xi, [1, 64 / batch_size]).round())
                )
                for j, pg in enumerate(self.optimizer.param_groups):
                    pg["lr"] = np.interp(
                        ni, xi,
                        [warmup_bias if j == 0 else 0.0,
                         pg["initial_lr"] * self.lf(epoch)],
                    )
                    if "momentum" in pg:
                        pg["momentum"] = np.interp(
                            ni, xi, [warmup_mom, momentum]
                        )

            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs   = self.model(images)
                loss_dict = self.criterion(
                    outputs=outputs, targets=targets, epoch=epoch
                )
                losses = loss_dict["losses"] * images.shape[0]

            self.scaler.scale(losses).backward()

            if ni - last_opt >= accumulate:
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                last_opt     = ni
                self._last_opt_step = last_opt

            if iter_i % self.log_interval == 0:
                t1     = time.time()
                n_pg   = len(self.optimizer.param_groups)
                cur_lr = self.optimizer.param_groups[min(2, n_pg - 1)]["lr"]
                log    = (f"[Epoch {epoch + 1}/{self.max_epoch}]"
                          f"[Iter {iter_i}/{epoch_size}]"
                          f"[lr {cur_lr:.6f}]")
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        log += f"[{k}: {float(v):.4f}]"
                    elif isinstance(v, bool):
                        log += f"[{k}: {v}]"
                log += f"[{t1 - t0:.2f}s]"
                print(log, flush=True)
                t0 = time.time()

    # ─────────────────────────── 主训练循环 ───────────────────────────

    def run(self, evaluator=None) -> float:
        """完整训练循环。

        Args:
            evaluator: ``Evaluator`` 实例；若为 None 则不执行验证。

        Returns:
            最终最优 mAP。
        """
        best_map = -1.0
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train_one_epoch(epoch)
            self.lr_scheduler.step()

            run_eval = (
                epoch % self.eval_interval == 0
                or epoch == self.max_epoch - 1
            )
            if run_eval and evaluator is not None:
                best_map = evaluator.evaluate(
                    self.model, self.val_loader, self.device, epoch, best_map
                )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return best_map

    @staticmethod
    def _resolve_collate_fn(dataset: BaseDataset):
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is None:
            raise TypeError(
                f"Dataset '{type(dataset).__name__}' must provide a collate_fn."
            )
        return collate_fn
