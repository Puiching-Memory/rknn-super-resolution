"""Distributed training context and process-group lifecycle."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Rank-aware helpers; all collectives go through this type."""

    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def broadcast_bool(self, value: bool, *, src: int = 0) -> bool:
        if self.world_size <= 1:
            return value
        flag = torch.tensor([int(value)], device=self.device)
        dist.broadcast(flag, src=src)
        return bool(flag.item())

    def all_reduce_avg(self, value: float) -> float:
        if self.world_size <= 1:
            return value
        tensor = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return float(tensor.item())

    def all_reduce_avg_dict(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.world_size <= 1:
            return metrics
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            reduced[key] = self.all_reduce_avg(value)
        return reduced


def _init_process_group(
    rank: int | None = None,
    world_size: int | None = None,
    *,
    device_id: torch.device | None = None,
) -> DistributedContext:
    rank_env = int(os.environ["RANK"]) if "RANK" in os.environ else None
    world_size_env = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else None

    if rank is not None and rank_env is not None and rank != rank_env:
        raise RuntimeError(f"Explicit rank {rank} does not match torchrun-provided RANK={rank_env}")
    if world_size is not None and world_size_env is not None and world_size != world_size_env:
        raise RuntimeError(
            f"Explicit world_size {world_size} does not match torchrun-provided "
            f"WORLD_SIZE={world_size_env}"
        )

    rank = rank_env if rank_env is not None else rank
    world_size = world_size_env if world_size_env is not None else world_size
    if rank is None or world_size is None:
        raise RuntimeError(
            "rank and world_size must be provided either by torchrun env vars or "
            "as arguments to distributed_session()"
        )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "MASTER_PORT is not set. When using torchrun it is provided by the launcher."
        )

    if device_id is None:
        device_id = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device_id)

    if world_size > 1:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device_id,
        )

    return DistributedContext(rank=rank, world_size=world_size, device=device_id)


@contextmanager
def distributed_session(
    rank: int | None = None,
    world_size: int | None = None,
    *,
    device_id: torch.device | None = None,
) -> Iterator[DistributedContext]:
    """Initialize NCCL (when world_size > 1) and yield a :class:`DistributedContext`."""
    ctx = _init_process_group(rank, world_size, device_id=device_id)
    try:
        yield ctx
    finally:
        if ctx.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
