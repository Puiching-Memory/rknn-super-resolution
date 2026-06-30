"""Unified step-based distributed training loop."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rk3588_mobile_sr.data.prefetch import BatchPrefetcher
from rk3588_mobile_sr.distributed.context import DistributedContext
from rk3588_mobile_sr.distributed.model import unwrap_model
from rk3588_mobile_sr.distributed.validation import (
    EarlyStopState,
    ValidationConfig,
    ValidationRunner,
)
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks
from rk3588_mobile_sr.utils.model_diagnostics import (
    ForwardDiagnosticsTracker,
    collect_training_diagnostics,
)
from rk3588_mobile_sr.utils.run_logger import logger
from rk3588_mobile_sr.utils.swanlab_logging import log_metrics
from rk3588_mobile_sr.utils.traceml_profiling import trace_training_step
from rk3588_mobile_sr.utils.train_framework import TrainAccel, amp_autocast, run_backward


def _unwrap_dataloader(train_loader: DataLoader | object) -> DataLoader | object:
    if hasattr(train_loader, "dataloader"):
        return train_loader.dataloader
    return train_loader


def _to_device(
    lr: torch.Tensor, hr: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if lr.device == device and hr.device == device:
        return lr, hr
    return lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)


def _format_component_detail(metrics: dict[str, float]) -> str:
    if "train/loss_charbonnier" not in metrics:
        return ""
    return (
        f" | charb={metrics['train/loss_charbonnier']:.4f}"
        f" | dct={metrics.get('train/loss_dct_weighted', 0.0):.4f}"
        f" | kd={metrics.get('train/loss_kd_weighted', 0.0):.4f}"
    )


class StepTrainer:
    """Single step-based trainer for all training stages."""

    def __init__(
        self,
        ctx: DistributedContext,
        model: nn.Module,
        train_loader: DataLoader | object,
        optimizer: optim.Optimizer,
        config: TrainConfig,
        hooks: TrainHooks,
        *,
        train_accel: TrainAccel | None = None,
        val_loader: DataLoader | None = None,
        save_dir: Path | None = None,
        validation_config: ValidationConfig | None = None,
        early_stop: EarlyStopState | None = None,
        model_diag: bool = True,
        global_step: int = 0,
    ) -> None:
        self.ctx = ctx
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.config = config
        self.hooks = hooks
        self.train_accel = train_accel or TrainAccel(
            enabled=False, dtype=torch.float32, scaler=None
        )
        self.val_loader = val_loader
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.validation_config = validation_config or ValidationConfig()
        self.early_stop = early_stop or EarlyStopState()
        self.model_diag = model_diag
        self.global_step = global_step
        self.unwrap = unwrap_model(model)

    def _log_plan(self) -> None:
        if not self.ctx.is_main:
            return
        plan = (
            "plan: max_steps={} val_every={} save_every={} log_every={}"
            + (" early_stop_patience={} min_delta={}" if self.early_stop.enabled else "")
        )
        if self.early_stop.enabled:
            logger.info(
                plan,
                self.config.max_steps,
                self.config.val_every,
                self.config.save_every,
                self.config.log_every,
                self.early_stop.patience,
                self.early_stop.min_delta,
            )
        else:
            logger.info(
                plan,
                self.config.max_steps,
                self.config.val_every,
                self.config.save_every,
                self.config.log_every,
            )

    def run(self) -> int:
        if self.save_dir is not None and self.ctx.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.barrier()
        self._log_plan()

        val_runner = ValidationRunner(
            ctx=self.ctx,
            model=self.model,
            val_loader=self.val_loader,
            config=self.validation_config,
            early_stop=self.early_stop,
            save_dir=self.save_dir,
            save_best_extra=self.hooks.save_best_extra,
            save_best=self.hooks.on_save_best,
        )

        raw_loader = _unwrap_dataloader(self.train_loader)
        prefetcher = BatchPrefetcher(raw_loader, buffer_size=self.config.prefetch_batches)
        prefetch_stream = torch.cuda.Stream(device=self.ctx.device)
        diag_tracker = (
            ForwardDiagnosticsTracker(self.model) if self.model_diag else None
        )

        window_loss = 0.0
        window_components: dict[str, float] = defaultdict(float)
        local_steps = 0
        pending_batch: tuple[torch.Tensor, torch.Tensor] | None = None

        def _fetch_batch() -> tuple[torch.Tensor, torch.Tensor]:
            lr, hr = next(prefetcher)
            with torch.cuda.stream(prefetch_stream):
                lr, hr = _to_device(lr, hr, self.ctx.device)
            return lr, hr

        try:
            pending_batch = _fetch_batch()
            while self.global_step < self.config.max_steps:
                torch.cuda.current_stream(self.ctx.device).wait_stream(prefetch_stream)
                lr, hr = pending_batch
                if self.global_step + 1 < self.config.max_steps:
                    pending_batch = _fetch_batch()

                with trace_training_step(self.model):
                    self.optimizer.zero_grad(set_to_none=True)
                    with amp_autocast(self.train_accel):
                        loss_result = self.hooks.loss_fn(self.unwrap, lr, hr)
                    if isinstance(loss_result, tuple):
                        loss, step_metrics = loss_result
                        if isinstance(step_metrics, dict):
                            for key, value in step_metrics.items():
                                window_components[key] += float(value)
                    else:
                        loss = loss_result
                    run_backward(loss, self.optimizer, self.train_accel)
                    if self.hooks.post_step is not None:
                        self.hooks.post_step(self.unwrap)
                    if self.hooks.scheduler is not None:
                        self.hooks.scheduler.step()

                self.global_step += 1
                local_steps += 1
                window_loss += loss.item()

                if self.hooks.on_step is not None:
                    self.hooks.on_step(self.global_step)

                if self.global_step % self.config.log_every == 0:
                    self._log_training_step(
                        window_loss,
                        local_steps,
                        window_components,
                        diag_tracker,
                    )
                    window_loss = 0.0
                    window_components.clear()
                    local_steps = 0

                if self.val_loader is not None and self.global_step % self.config.val_every == 0:
                    val_result = val_runner.run(self.global_step)
                    if val_result is not None and val_result.should_stop:
                        break

                if (
                    self.ctx.is_main
                    and self.save_dir is not None
                    and self.global_step % self.config.save_every == 0
                ):
                    step_path = self.save_dir / f"step_{self.global_step}.pth"
                    if self.hooks.on_save_step is not None:
                        self.hooks.on_save_step(self.global_step, step_path)
                    else:
                        torch.save(self.unwrap.state_dict(), step_path)
        finally:
            prefetcher.close()
            if diag_tracker is not None:
                diag_tracker.close()

        if self.ctx.is_main and self.save_dir is not None:
            last_path = self.save_dir / "last.pth"
            if self.hooks.on_save_last is not None:
                self.hooks.on_save_last(self.global_step, last_path)
            else:
                torch.save(self.unwrap.state_dict(), last_path)

        return self.global_step

    def _log_training_step(
        self,
        window_loss: float,
        local_steps: int,
        window_components: dict[str, float],
        diag_tracker: ForwardDiagnosticsTracker | None,
    ) -> None:
        avg_loss = self.ctx.all_reduce_avg(window_loss / local_steps)
        reduced: dict[str, float] = {"train/loss": avg_loss}
        if self.hooks.scheduler is not None:
            reduced["train/lr"] = self.hooks.scheduler.get_last_lr()[0]
        elif self.ctx.is_main:
            reduced["train/lr"] = self.optimizer.param_groups[0]["lr"]
        for key, value in window_components.items():
            reduced[key] = self.ctx.all_reduce_avg(value / local_steps)

        if self.ctx.is_main:
            if self.hooks.log_train_metrics is not None:
                self.hooks.log_train_metrics(reduced, self.global_step)
            else:
                log_metrics(reduced, step=self.global_step)
                lr_note = ""
                if "train/lr" in reduced:
                    lr_note = f" | lr={reduced['train/lr']:.6f}"
                logger.info(
                    "Step {} | loss={:.4f}{}{}",
                    self.global_step,
                    reduced["train/loss"],
                    _format_component_detail(reduced),
                    lr_note,
                )

        if diag_tracker is not None:
            diag = collect_training_diagnostics(self.model, diag_tracker)
            diag = self.ctx.all_reduce_avg_dict(diag)
            if self.ctx.is_main:
                log_metrics(diag, step=self.global_step)
