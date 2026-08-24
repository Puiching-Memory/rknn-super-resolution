"""Training session bootstrap (logging, experiment tracking, loaders)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from rknn_super_resolution.distributed.context import DistributedContext
from rknn_super_resolution.distributed.validation import EarlyStopState, ValidationConfig
from rknn_super_resolution.train.loop import StepTrainer
from rknn_super_resolution.train.types import LoaderBundle, TrainConfig, TrainHooks
from rknn_super_resolution.utils.run_logger import logger, setup_run_logger
from rknn_super_resolution.utils.swanlab_logging import (
    build_swanlab_run_config,
    finish_swanlab,
    resolve_swanlab_run_id,
    setup_swanlab,
)
from rknn_super_resolution.utils.traceml_profiling import finish_traceml, setup_traceml
from rknn_super_resolution.utils.train_framework import TrainAccel, build_loaders


class TrainSession:
    """Shared DDP training bootstrap for all stages."""

    def __init__(
        self,
        ctx: DistributedContext,
        args: argparse.Namespace,
        *,
        save_dir: Path | str,
        experiment_name: str,
    ) -> None:
        self.ctx = ctx
        self.args = args
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name

    def prepare(self) -> None:
        """Create save dir, logger, SwanLab, TraceML."""
        if self.ctx.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.barrier()
        setup_run_logger(self.save_dir, self.ctx.rank)
        experiment_name = self.args.swanlab_experiment or self.experiment_name
        resume_checkpoint = getattr(self.args, "resume", None)
        swanlab_run_id = resolve_swanlab_run_id(
            save_dir=self.save_dir,
            project=self.args.swanlab_project,
            experiment_name=experiment_name,
            resume_checkpoint=resume_checkpoint,
            explicit_run_id=getattr(self.args, "swanlab_run_id", None),
        )
        setup_swanlab(
            rank=self.ctx.rank,
            save_dir=self.save_dir,
            project=self.args.swanlab_project,
            experiment_name=experiment_name,
            config=build_swanlab_run_config(
                self.args,
                world_size=self.ctx.world_size,
                vmaf_enc_size=ValidationConfig().vmaf_enc_size,
            ),
            disabled=self.args.no_swanlab,
            resume_training=resume_checkpoint is not None,
            run_id=swanlab_run_id,
            resume="must",
        )
        setup_traceml(self.args, rank=self.ctx.rank)

    def build_loaders(
        self,
        *,
        train_aug: bool = True,
        val_bs: int = 1,
    ) -> LoaderBundle:
        train_loader, _, val_loader = build_loaders(
            self.args,
            self.ctx.device,
            train_aug=train_aug,
            val_bs=val_bs,
            distributed=True,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
        )
        return LoaderBundle(train=train_loader, val=val_loader)

    def run_trainer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loaders: LoaderBundle,
        config: TrainConfig,
        hooks: TrainHooks,
        *,
        train_accel: TrainAccel | None = None,
        validation_config: ValidationConfig | None = None,
        early_stop: EarlyStopState | None = None,
        model_diag: bool = True,
        global_step: int = 0,
        save_dir: Path | str | None = None,
    ) -> int:
        trainer = StepTrainer(
            self.ctx,
            model,
            loaders.train,
            optimizer,
            config,
            hooks,
            train_accel=train_accel,
            val_loader=loaders.val,
            save_dir=Path(save_dir) if save_dir is not None else self.save_dir,
            validation_config=validation_config,
            early_stop=early_stop,
            model_diag=model_diag,
            global_step=global_step,
        )
        return trainer.run()

    def finalize(self) -> None:
        finish_traceml(rank=self.ctx.rank)
        finish_swanlab()

    def log_config(self, **fields: object) -> None:
        if self.ctx.is_main:
            parts = " ".join(f"{k}={{{i}}}" for i, k in enumerate(fields))
            logger.info("config: " + parts, *fields.values())
