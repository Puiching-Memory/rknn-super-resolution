"""Training loop types and configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    log_every: int = 500
    val_every: int = 1000
    save_every: int = 5000
    prefetch_batches: int = 4
    val_scale: int = 3


@dataclass
class TrainHooks:
    """Stage-specific callbacks injected into :class:`StepTrainer`."""

    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor | tuple]
    scheduler: optim.lr_scheduler._LRScheduler | None = None
    on_step: Callable[[int], None] | None = None
    post_step: Callable[[nn.Module], None] | None = None
    save_best_extra: Callable[[Path], None] | None = None
    on_save_step: Callable[[int, Path], None] | None = None
    on_save_best: Callable[[Path, int], None] | None = None
    on_save_last: Callable[[int, Path], None] | None = None
    log_train_metrics: Callable[[dict[str, float], int], None] | None = None
    on_validation: Callable[[object], None] | None = None


@dataclass
class LoaderBundle:
    train: object
    val: object | None = None
