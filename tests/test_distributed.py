"""Tests for distributed training primitives (world_size=1 + mocks)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rknn_super_resolution.distributed.context import DistributedContext, distributed_session
from rknn_super_resolution.distributed.model import wrap_training_model
from rknn_super_resolution.distributed.sync import rank0_section
from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.train.loop import StepTrainer
from rknn_super_resolution.train.types import TrainConfig, TrainHooks
from rknn_super_resolution.train.validation import ValidationResult
from rknn_super_resolution.utils.train_framework import (
    load_training_module_state_dict,
    training_module_state_dict,
)


def test_rank0_section_main_runs_fn_then_barriers():
    ctx = DistributedContext(rank=0, world_size=2, device=torch.device("cpu"))
    ran: list[bool] = []

    with patch("rknn_super_resolution.distributed.context.dist.barrier") as mock_barrier:
        rank0_section(ctx, lambda: ran.append(True))

    assert ran == [True]
    mock_barrier.assert_called_once()


def test_rank0_section_non_main_skips_fn_still_barriers():
    ctx = DistributedContext(rank=1, world_size=2, device=torch.device("cpu"))
    ran: list[bool] = []

    with patch("rknn_super_resolution.distributed.context.dist.barrier") as mock_barrier:
        rank0_section(ctx, lambda: ran.append(True))

    assert ran == []
    mock_barrier.assert_called_once()


def test_distributed_context_world_size_one_no_collectives():
    ctx = DistributedContext(rank=0, world_size=1, device=torch.device("cpu"))
    assert ctx.is_main
    assert ctx.all_reduce_avg(3.5) == 3.5
    assert ctx.broadcast_bool(True) is True
    ctx.barrier()  # no-op


def test_step_trainer_runs_steps_and_validation(tmp_path):
    device = torch.device("cpu")
    ctx = DistributedContext(rank=0, world_size=1, device=device)
    model = PhaseRLFNSR(num_channels=8, num_blocks=1, scale=3).to(device)
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
        objective=nn.functional.l1_loss,
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
            score=30.0,
            val_metrics=None,
            improved=True,
            should_stop=step >= 10,
        )

    with patch("rknn_super_resolution.train.loop.ValidationRunner.run", fake_run):
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


def test_step_trainer_never_bypasses_wrapped_model(tmp_path):
    """A wrapper exposing ``.module`` must still own every training forward."""

    class _ForwardGuard(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = PhaseRLFNSR(num_channels=4, num_blocks=1)
            self.forward_calls = 0

        def forward(self, *args, **kwargs):
            self.forward_calls += 1
            return self.module(*args, **kwargs)

    device = torch.device("cpu")
    ctx = DistributedContext(rank=0, world_size=1, device=device)
    model = _ForwardGuard()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = torch.rand(1, 3, 8, 8) * 255.0
    hr = torch.rand(1, 3, 24, 24) * 255.0

    class _InfiniteLoader:
        def __iter__(self):
            while True:
                yield lr, hr

    trainer = StepTrainer(
        ctx,
        model,
        _InfiniteLoader(),
        optimizer,
        TrainConfig(
            max_steps=1,
            log_every=1,
            val_every=10,
            save_every=10,
            prefetch_batches=1,
        ),
        TrainHooks(objective=nn.functional.l1_loss),
        save_dir=tmp_path,
        model_diag=False,
    )
    trainer.run()
    assert model.forward_calls == 1


def test_torch_state_dict_api_canonicalizes_compiled_model() -> None:
    model = torch.compile(nn.Linear(3, 2))
    state = training_module_state_dict(model)
    assert set(state) == {"weight", "bias"}
    replacement = {name: torch.zeros_like(value) for name, value in state.items()}
    load_training_module_state_dict(model, replacement)
    assert all(torch.count_nonzero(value) == 0 for value in model.parameters())


def test_cuda_barrier_passes_local_device_index():
    ctx = DistributedContext(rank=3, world_size=4, device=torch.device("cuda:1"))
    with patch("rknn_super_resolution.distributed.context.dist.barrier") as mock_barrier:
        ctx.barrier()
    mock_barrier.assert_called_once_with(device_ids=[1])


def test_ddp_wrap_uses_device_index_not_global_rank():
    captured: dict[str, object] = {}

    class _FakeDDP(nn.Module):
        def __init__(self, module, device_ids=None, output_device=None, **kwargs):
            super().__init__()
            captured["device_ids"] = device_ids
            captured["output_device"] = output_device
            self.module = module

    ctx = DistributedContext(rank=3, world_size=4, device=torch.device("cuda:1"))
    model = nn.Linear(2, 2)
    with patch("rknn_super_resolution.distributed.model.DDP", _FakeDDP):
        wrapped = wrap_training_model(model, ctx, compile_model=False)
    assert captured["device_ids"] == [1]
    assert captured["output_device"] == 1
    assert wrapped.module is model


def test_distributed_session_requires_torchrun_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(RuntimeError, match="torchrun"), distributed_session():
        pass
