"""Quick multi-GPU canvas codec training benchmark via torchrun."""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from rk3588_mobile_sr.utils.train_framework import (
    build_loaders,
    build_model,
    cleanup_ddp,
    setup_ddp,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_manifest", default="data/codec_cache/manifest.jsonl")
    parser.add_argument("--decode", default="auto", choices=["auto", "dali"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    ns = argparse.Namespace(
        config=None,
        codec_manifest=args.codec_manifest,
        decode=args.decode,
        val_manifest=None,
        scale=3,
        num_channels=32,
        num_blocks=8,
        num_conv_branches=4,
        negative_slope=0.1,
        colorspace="yuv",
        batch_size=args.batch_size,
    )

    model = build_model(ns, device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    train_loader, _, _ = build_loaders(
        ns, device, train_aug=True, distributed=True, rank=rank, world_size=world_size
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def run_steps(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data_iter = iter(train_loader.dataloader)
        for _ in range(n):
            lr, hr = next(data_iter)
            if lr.device != device:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(lr)
            loss = criterion(out, hr)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        run_steps(1)

    elapsed = run_steps(args.steps)
    global_batch = args.batch_size * world_size
    img_per_s = args.steps * global_batch / elapsed

    if rank == 0:
        print(
            f"[codec canvas] {world_size} GPUs x batch {args.batch_size} | "
            f"decode={args.decode} | "
            f"{args.steps} steps in {elapsed:.2f}s | "
            f"{args.steps / elapsed:.2f} step/s | {img_per_s:.0f} img/s"
        )

    train_loader.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
