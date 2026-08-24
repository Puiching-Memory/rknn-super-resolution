"""DDP synchronization primitives."""

from __future__ import annotations

from collections.abc import Callable

from rknn_super_resolution.distributed.context import DistributedContext


def rank0_section(ctx: DistributedContext, fn: Callable[[], None]) -> None:
    """Run ``fn`` on rank 0, then barrier all ranks (prevents post-collective deadlock)."""
    if ctx.is_main:
        fn()
    ctx.barrier()
