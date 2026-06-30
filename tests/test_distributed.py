"""Tests for distributed training primitives (world_size=1 + mocks)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rk3588_mobile_sr.distributed.context import DistributedContext
from rk3588_mobile_sr.distributed.sync import rank0_section
from rk3588_mobile_sr.distributed.validation import EarlyStopState, ValidationResult
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.train.loop import StepTrainer
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks


def test_rank0_section_main_runs_fn_then_barriers():
    ctx = DistributedContext(rank=0, world_size=2, device=torch.device("cpu"))
    ran: list[bool] = []

    with patch("rk3588_mobile_sr.distributed.context.dist.barrier") as mock_barrier:
        rank0_section(ctx, lambda: ran.append(True))

    assert ran == [True]
    mock_barrier.assert_called_once()


def test_rank0_section_non_main_skips_fn_still_barriers():
    ctx = DistributedContext(rank=1, world_size=2, device=torch.device("cpu"))
    ran: list[bool] = []

    with patch("rk3588_mobile_sr.distributed.context.dist.barrier") as mock_barrier:
        rank0_section(ctx, lambda: ran.append(True))

    assert ran == []
    mock_barrier.assert_called_once()


def test_early_stop_improves_resets_patience():
    state = EarlyStopState(enabled=True, patience=3, min_delta=0.1)
    improved, stop = state.update(25.0)
    assert improved and not stop
    assert state.best_psnr == 25.0
    assert state.patience_counter == 0

    improved, stop = state.update(25.5)
    assert improved and not stop
    assert state.patience_counter == 0


def test_early_stop_triggers_after_patience():
    state = EarlyStopState(enabled=True, patience=2, min_delta=0.1)
    state.update(20.0)

    _, stop = state.update(20.05)
    assert not stop
    assert state.patience_counter == 1

    _, stop = state.update(20.05)
    assert stop
    assert state.patience_counter == 2


def test_early_stop_disabled_never_stops():
    state = EarlyStopState(enabled=False, patience=1, min_delta=0.1)
    for psnr in (10.0, 9.0, 8.0):
        _, stop = state.update(psnr)
        assert not stop


def test_distributed_context_world_size_one_no_collectives():
    ctx = DistributedContext(rank=0, world_size=1, device=torch.device("cpu"))
    assert ctx.is_main
    assert ctx.all_reduce_avg(3.5) == 3.5
    assert ctx.broadcast_bool(True) is True
    ctx.barrier()  # no-op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="StepTrainer uses CUDA streams")
def test_step_trainer_runs_steps_and_validation(tmp_path):
    device = torch.device("cuda:0")
    ctx = DistributedContext(rank=0, world_size=1, device=device)
    model = MobileOneSR(num_channels=8, num_blocks=1, scale=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 96, 96) * 255.0
    base_loader = DataLoader(TensorDataset(lr, hr), batch_size=2, shuffle=True)
    val_loader = DataLoader(TensorDataset(lr[:2], hr[:2]), batch_size=1)

    class _CycleLoader:
        def __iter__(self):
            while True:
                yield from base_loader

    train_loader = _CycleLoader()

    hooks = TrainHooks(
        loss_fn=lambda m, lr_t, hr_t: nn.functional.l1_loss(m(lr_t), hr_t),
    )
    config = TrainConfig(
        max_steps=10,
        log_every=5,
        val_every=5,
        save_every=100,
        prefetch_batches=1,
    )

    val_steps: list[int] = []

    def fake_run(_self, step: int) -> ValidationResult:
        val_steps.append(step)
        return ValidationResult(
            step=step,
            psnr=30.0,
            val_metrics=None,
            improved=True,
            should_stop=step >= 10,
        )

    with patch("rk3588_mobile_sr.train.loop.ValidationRunner.run", fake_run):
        trainer = StepTrainer(
            ctx,
            model,
            train_loader,
            optimizer,
            config,
            hooks,
            val_loader=val_loader,
            save_dir=tmp_path,
            model_diag=False,
        )
        final_step = trainer.run()

    assert final_step == 10
    assert val_steps == [5, 10]
    assert (tmp_path / "last.pth").exists()
