"""Distributed validation with rank-0 post-processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rknn_super_resolution.distributed.context import DistributedContext
from rknn_super_resolution.distributed.model import is_compiled_module
from rknn_super_resolution.distributed.sync import rank0_section
from rknn_super_resolution.utils.model_diagnostics import check_deploy_consistency
from rknn_super_resolution.utils.run_logger import logger
from rknn_super_resolution.utils.sr_metrics import ValidationMetrics, validate_ddp_extended
from rknn_super_resolution.utils.swanlab_logging import log_metrics, log_validation_sr_images
from rknn_super_resolution.utils.train_framework import training_module_state_dict
from rknn_super_resolution.utils.vmaf_metric import DEFAULT_VMAF_MODEL


@dataclass
class ValidationConfig:
    scale: int = 3
    extended: bool = True
    compute_vmaf: bool = True
    vmaf_model: str = DEFAULT_VMAF_MODEL
    # Encode-side CAMBI size (H, W) for VMAF v1; LR canvas before SR display.
    vmaf_enc_size: tuple[int, int] | None = (360, 640)
    log_images: bool = True
    deploy_check: bool = True
    vis_samples: int = 8
    vis_max_size: int = 768
    colorspace: str = "yuv"
    data_preview: bool = True
    final_preview: bool = True


@dataclass
class EarlyStopState:
    enabled: bool = False
    patience: int = 8
    min_delta: float = 0.1
    min_evaluations: int = 0
    best_score: float = -1.0
    plateau_score: float = -1.0
    psnr_at_best: float = -1.0
    patience_counter: int = 0
    evaluations: int = 0

    def update(self, score: float) -> tuple[bool, bool]:
        """Return (improved, should_stop). Higher score is better (VMAF / PSNR)."""
        self.evaluations += 1
        if not self.enabled:
            improved = score > self.best_score
            if improved:
                self.best_score = score
            return improved, False

        improved = score > self.best_score
        if improved:
            self.best_score = score

        significant = score > self.plateau_score + self.min_delta
        if significant:
            self.plateau_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        should_stop = (
            self.evaluations >= self.min_evaluations and self.patience_counter >= self.patience
        )
        return improved, should_stop


def primary_metric_logs(
    *,
    primary_key: str,
    best_score: float,
    psnr_at_best: float,
) -> dict[str, float]:
    """SwanLab keys for the tracked best checkpoint.

    ``val/best_score`` is the primary metric (VMAF or PSNR). ``val/best_psnr``
    is always the PSNR of that checkpoint, not an alias of VMAF.
    """
    metrics = {"val/best_score": best_score}
    if psnr_at_best >= 0.0:
        metrics["val/best_psnr"] = psnr_at_best
    if primary_key == "val/vmaf":
        metrics["val/best_vmaf"] = best_score
    return metrics


@dataclass
class ValidationResult:
    step: int
    score: float
    val_metrics: ValidationMetrics | None
    improved: bool
    should_stop: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationRunner:
    ctx: DistributedContext
    model: nn.Module
    val_loader: DataLoader | None
    config: ValidationConfig
    early_stop: EarlyStopState
    save_dir: Path | None = None
    save_best_extra: Callable[[Path, int], None] | None = None
    save_best: Callable[[Path, int], None] | None = None

    def run(self, step: int) -> ValidationResult | None:
        if self.val_loader is None:
            return None

        if self.ctx.is_main:
            logger.info("Step {} validation started", step)

        score, val_metrics = validate_ddp_extended(
            self.model,
            self.val_loader,
            self.ctx.rank,
            self.ctx.world_size,
            scale=self.config.scale,
            compute_vmaf=self.config.compute_vmaf,
            vmaf_model=self.config.vmaf_model,
            vmaf_enc_size=self.config.vmaf_enc_size,
            colorspace=self.config.colorspace,
        )

        improved, should_stop = self.early_stop.update(score)
        result = ValidationResult(
            step=step,
            score=score,
            val_metrics=val_metrics,
            improved=improved,
            should_stop=should_stop,
        )

        def _main_hooks() -> None:
            if improved and val_metrics is not None:
                self.early_stop.psnr_at_best = val_metrics.psnr
            primary_key = (
                "val/vmaf"
                if self.config.compute_vmaf
                and val_metrics is not None
                and val_metrics.vmaf is not None
                else "val/psnr"
            )
            metrics: dict[str, float] = primary_metric_logs(
                primary_key=primary_key,
                best_score=self.early_stop.best_score,
                psnr_at_best=self.early_stop.psnr_at_best,
            )
            if val_metrics is not None:
                metrics.update(val_metrics.to_log_dict())
            else:
                metrics[primary_key] = score
            if self.early_stop.enabled:
                metrics["early_stop/patience"] = self.early_stop.patience_counter

            if self.config.deploy_check and not is_compiled_module(self.model):
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
                    torch.save(training_module_state_dict(self.model), best_path)
                if self.save_best_extra is not None:
                    self.save_best_extra(best_path, step)

            result.metrics = metrics

        rank0_section(self.ctx, _main_hooks)
        should_stop = self.ctx.broadcast_bool(should_stop)
        result.should_stop = should_stop
        return result
