"""torchrun-launched NCCL process group and rank helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size <= 1:
            return
        if self.device.type == "cuda" and self.device.index is not None:
            dist.barrier(device_ids=[self.device.index])
        else:
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
        return {key: self.all_reduce_avg(value) for key, value in metrics.items()}


@contextmanager
def distributed_session() -> Iterator[DistributedContext]:
    """Bind this process to ``LOCAL_RANK`` and initialize NCCL."""
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as exc:
        raise RuntimeError(
            "launch with torchrun so RANK, WORLD_SIZE, and LOCAL_RANK are set"
        ) from exc

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        yield DistributedContext(rank=rank, world_size=world_size, device=device)
    finally:
        dist.destroy_process_group()
