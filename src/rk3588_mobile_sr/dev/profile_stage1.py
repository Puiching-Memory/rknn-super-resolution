"""Profile stage-1 training step: data vs compute breakdown."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from rk3588_mobile_sr.data.prefetch import BatchPrefetcher
from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.utils.train_framework import build_train_accel, maybe_compile, run_backward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_manifest", default="data/codec_cache/manifest.jsonl")
    parser.add_argument("--decode", default="auto", choices=["auto", "dali"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr_size", default="360,640")
    parser.add_argument("--hr_size", default="1080,1920")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)
    lr_h, lr_w = (int(x) for x in args.lr_size.split(","))
    hr_h, hr_w = (int(x) for x in args.hr_size.split(","))

    settings = TrainDataSettings(
        codec_manifest=args.codec_manifest,
        lr_size=(lr_h, lr_w),
        hr_size=(hr_h, hr_w),
        decode=args.decode,
    )
    bundle = build_codec_train_loader(
        settings,
        batch_size=args.batch_size,
        rank=0,
        seed=0,
        device_id=args.device_id,
    )
    prefetcher = BatchPrefetcher(bundle.dataloader, buffer_size=args.prefetch_batches)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    data_ms: list[float] = []
    for i in range(args.warmup + args.steps):
        start.record()
        lr, hr = next(prefetcher)
        if lr.device != device:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            data_ms.append(start.elapsed_time(end))

    model = MobileOneSR().to(device)
    if args.compile:
        model = maybe_compile(model, argparse.Namespace(compile=True))
    model.train()
    accel = build_train_accel(argparse.Namespace(amp=True))
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = torch.randn(args.batch_size, 3, lr_h, lr_w, device=device)
    hr = torch.randn(args.batch_size, 3, hr_h, hr_w, device=device)
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
    bundle.close()
    avg_data = sum(data_ms) / len(data_ms)
    avg_compute = sum(compute_ms) / len(compute_ms)
    ideal = max(avg_data, avg_compute)
    img_per_s = args.batch_size * 1000.0 / ideal

    print(f"batch_size={args.batch_size} compile={args.compile} decode={args.decode}")
    print(f"  data+prefetch: {avg_data:.1f} ms/batch")
    print(f"  compute only:  {avg_compute:.1f} ms/step")
    print(f"  ideal overlap: {ideal:.1f} ms/step -> {img_per_s:.0f} img/s per GPU")


if __name__ == "__main__":
    main()
