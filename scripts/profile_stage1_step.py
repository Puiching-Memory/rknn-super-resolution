"""Profile stage-1 training step: data vs compute breakdown."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from data.div2k_dali import build_dali_train_loader
from data.div2k_lmdb import build_lmdb_train_loader
from data.prefetch import BatchPrefetcher
from models.mobileone_sr import MobileOneSR
from utils.train_framework import build_train_accel, maybe_compile, run_backward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--dali_num_threads", type=int, default=16)
    parser.add_argument("--lmdb_dir", type=str, default=None, help="LMDB patch cache directory")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

    if args.lmdb_dir:
        loader, _ = build_lmdb_train_loader(
            args.lmdb_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            distributed=False,
        )
        backend = "lmdb"
    else:
        loader = build_dali_train_loader(
            "data/DIV2K_train_HR",
            "data/DIV2K_train_LR_bicubic/X3",
            scale=3,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device_id=args.device_id,
            shard_id=0,
            num_shards=1,
            num_threads=args.dali_num_threads,
            samples_per_image=2,
        )
        backend = "dali"
    prefetcher = BatchPrefetcher(loader, buffer_size=args.prefetch_batches)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    data_ms: list[float] = []
    for _ in range(args.warmup + args.steps):
        start.record()
        lr, hr = next(prefetcher)
        if lr.device != device:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        if _ >= args.warmup:
            data_ms.append(start.elapsed_time(end))

    model = MobileOneSR().to(device)
    if args.compile:
        model = maybe_compile(model, argparse.Namespace(compile=True))
    model.train()
    accel = build_train_accel(argparse.Namespace(amp=True))
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = torch.randn(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
    hr = torch.randn(
        args.batch_size,
        3,
        args.patch_size * 3,
        args.patch_size * 3,
        device=device,
    )
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=accel.dtype):
            loss = criterion(model(lr), hr)
        run_backward(loss, optimizer, accel)

    compute_ms: list[float] = []
    for _ in range(args.steps):
        start.record()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=accel.dtype):
            loss = criterion(model(lr), hr)
        run_backward(loss, optimizer, accel)
        end.record()
        torch.cuda.synchronize()
        compute_ms.append(start.elapsed_time(end))

    prefetcher.close()
    avg_data = sum(data_ms) / len(data_ms)
    avg_compute = sum(compute_ms) / len(compute_ms)
    ideal = max(avg_data, avg_compute)
    img_per_s = args.batch_size * 1000.0 / ideal

    print(f"batch_size={args.batch_size} compile={args.compile} backend={backend}")
    print(f"  data+prefetch: {avg_data:.1f} ms/batch")
    print(f"  compute only:  {avg_compute:.1f} ms/step")
    print(f"  ideal overlap: {ideal:.1f} ms/step -> {img_per_s:.0f} img/s per GPU")
    print(f"  GPU duty cycle (compute/ideal): {100 * avg_compute / ideal:.0f}%")
    print("Note: LMDB reads pre-decoded uint8 patches; DALI decodes JPEG at train time.")


if __name__ == "__main__":
    main()
