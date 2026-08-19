"""Distributed validation with rank-0 post-processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rk3588_mobile_sr.distributed.context import DistributedContext
from rk3588_mobile_sr.distributed.model import is_compiled_module, unwrap_model
from rk3588_mobile_sr.distributed.sync import rank0_section
from rk3588_mobile_sr.utils.model_diagnostics import check_deploy_consistency
from rk3588_mobile_sr.utils.run_logger import logger
from rk3588_mobile_sr.utils.sr_metrics import ValidationMetrics, validate_ddp, validate_ddp_extended
from rk3588_mobile_sr.utils.swanlab_logging import log_metrics, log_validation_sr_images
from rk3588_mobile_sr.utils.vmaf_metric import DEFAULT_VMAF_MODEL


@dataclass
class ValidationConfig:
    scale: int = 3
    extended: bool = True
    compute_dists: bool = False
    compute_vmaf: bool = True
    vmaf_model: str = DEFAULT_VMAF_MODEL
    # Encode-side CAMBI size (H, W) for VMAF v1; LR canvas before SR display.
    vmaf_enc_size: tuple[int, int] | None = (360, 640)
    log_images: bool = True
    deploy_check: bool = True
    vis_samples: int = 8
    vis_max_size: int = 768
    colorspace: str = "rgb"
    data_preview: bool = True


@dataclass
class EarlyStopState:
    enabled: bool = False
    patience: int = 8
    min_delta: float = 0.1
    best_score: float = -1.0
    patience_counter: int = 0

    # Backward-compatible alias used by older tests / logs.
    @property
    def best_psnr(self) -> float:
        return self.best_score

    @best_psnr.setter
    def best_psnr(self, value: float) -> None:
        self.best_score = value

    def update(self, score: float) -> tuple[bool, bool]:
        """Return (improved, should_stop). Higher score is better (VMAF / PSNR)."""
        if not self.enabled:
            improved = score > self.best_score
            if improved:
                self.best_score = score
            return improved, False

        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        should_stop = self.patience_counter >= self.patience
        return improved, should_stop


@dataclass
class ValidationResult:
    step: int
    score: float
    val_metrics: ValidationMetrics | None
    improved: bool
    should_stop: bool
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def psnr(self) -> float:
        """Legacy alias: primary score (VMAF when enabled)."""
        return self.score


@dataclass
class ValidationRunner:
    ctx: DistributedContext
    model: nn.Module
    val_loader: DataLoader | None
    config: ValidationConfig
    early_stop: EarlyStopState
    save_dir: Path | None = None
    save_best_extra: Callable[[Path], None] | None = None
    save_best: Callable[[Path, int], None] | None = None

    def run(self, step: int) -> ValidationResult | None:
        if self.val_loader is None:
            return None

        if self.ctx.is_main:
            logger.info("Step {} validation started", step)

        if self.config.extended:
            score, val_metrics = validate_ddp_extended(
                self.model,
                self.val_loader,
                self.ctx.rank,
                self.ctx.world_size,
                scale=self.config.scale,
                compute_dists=self.config.compute_dists,
                compute_vmaf=self.config.compute_vmaf,
                vmaf_model=self.config.vmaf_model,
                vmaf_enc_size=self.config.vmaf_enc_size,
                colorspace=self.config.colorspace,
            )
        else:
            score = validate_ddp(
                self.model,
                self.val_loader,
                self.ctx.rank,
                self.ctx.world_size,
                scale=self.config.scale,
            )
            val_metrics = None

        improved, should_stop = self.early_stop.update(score)
        result = ValidationResult(
            step=step,
            score=score,
            val_metrics=val_metrics,
            improved=improved,
            should_stop=should_stop,
        )

        def _main_hooks() -> None:
            primary_key = (
                "val/vmaf"
                if self.config.compute_vmaf and val_metrics is not None and val_metrics.vmaf is not None
                else "val/psnr"
            )
            metrics: dict[str, float] = {
                "val/best_score": self.early_stop.best_score,
                "val/best_psnr": self.early_stop.best_score,  # legacy SwanLab key
            }
            if val_metrics is not None:
                metrics.update(val_metrics.to_log_dict())
            else:
                metrics[primary_key] = score
            if self.early_stop.enabled:
                metrics["early_stop/patience"] = self.early_stop.patience_counter

            unwrap = unwrap_model(self.model)
            if self.config.deploy_check and not is_compiled_module(unwrap):
                deploy_metrics = check_deploy_consistency(
                    self.model, self.val_loader, self.ctx.device
                )
                if deploy_metrics:
                    metrics.update(deploy_metrics)
            elif self.config.deploy_check:
                logger.info("skipping deploy check (torch.compile)")

            log_metrics(metrics, step=step)
            if self.config.log_images:
                try:
                    log_validation_sr_images(
                        self.model,
                        self.val_loader,
                        self.ctx.device,
                        step=step,
                        num_samples=self.config.vis_samples,
                        max_size=self.config.vis_max_size,
                        colorspace=self.config.colorspace,
                    )
                except Exception as exc:
                    logger.warning("validation image upload failed: {}", exc)

            stop_msg = " | early stop" if should_stop else ""
            detail = ""
            if val_metrics is not None:
                detail = (
                    f" | PSNR={val_metrics.psnr:.2f} | Y-PSNR={val_metrics.y_psnr:.2f} "
                    f"| SSIM={val_metrics.ssim:.4f}"
                )
                if val_metrics.vmaf is not None:
                    detail = f" | VMAF={val_metrics.vmaf:.2f}" + detail
                if val_metrics.dists is not None:
                    detail += f" | DISTS={val_metrics.dists:.4f}"
            patience_note = ""
            if self.early_stop.enabled:
                patience_note = (
                    f" | patience={self.early_stop.patience_counter}/{self.early_stop.patience}"
                )
            label = "VMAF" if primary_key == "val/vmaf" else "PSNR"
            logger.info(
                "Step {} | val {}={:.2f}{} | best={:.2f}{}{}",
                step,
                label,
                score,
                detail,
                self.early_stop.best_score,
                patience_note,
                stop_msg,
            )

            if improved and self.save_dir is not None:
                best_path = self.save_dir / "best.pth"
                if self.save_best is not None:
                    self.save_best(best_path, step)
                else:
                    torch.save(unwrap.state_dict(), best_path)
                if self.save_best_extra is not None:
                    self.save_best_extra(best_path)

            result.metrics = metrics

        rank0_section(self.ctx, _main_hooks)
        should_stop = self.ctx.broadcast_bool(should_stop)
        result.should_stop = should_stop
        return result
