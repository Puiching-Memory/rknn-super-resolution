"""Training loop types and configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    log_every: int = 500
    val_every: int = 1000
    save_every: int = 5000
    prefetch_batches: int = 4


@dataclass
class TrainHooks:
    """Stage callbacks that cannot access or invoke the training model."""

    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    post_step: Callable[[], None] | None = None
    save_best_extra: Callable[[Path, int], None] | None = None
    on_save_step: Callable[[int, Path], None] | None = None
    on_save_best: Callable[[Path, int], None] | None = None
    on_save_last: Callable[[int, Path], None] | None = None
    on_validation: Callable[[object], None] | None = None


@dataclass
class LoaderBundle:
    train: object
    val: object | None = None
