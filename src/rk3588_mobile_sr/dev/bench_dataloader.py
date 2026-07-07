"""Benchmark canvas codec data loading throughput."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from rk3588_mobile_sr.data.prefetch import BatchPrefetcher
from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR


def bench_loader(
    name: str,
    loader,
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    warmup: int,
    with_model: bool,
    prefetch_batches: int,
) -> dict[str, float]:
    model = None
    criterion = None
    optimizer = None
    if with_model:
        model = MobileOneSR().to(device)
        model.train()
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    prefetcher = BatchPrefetcher(loader, buffer_size=prefetch_batches)

    def run_steps(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            lr, hr = next(prefetcher)
            if lr.device.type != "cuda" or lr.device.index != device.index:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
            if with_model:
                optimizer.zero_grad(set_to_none=True)
                out = model(lr)
                loss = criterion(out, hr)
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        run_steps(1)
    elapsed = run_steps(steps)
    prefetcher.close()
    return {
        "name": name,
        "steps": steps,
        "elapsed_s": elapsed,
        "step_per_s": steps / elapsed,
        "img_per_s": steps * batch_size / elapsed,
        "batch_size": batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_manifest", default="data/codec_cache/manifest.jsonl")
    parser.add_argument("--decode", default="auto", choices=["auto", "dali", "torchcodec"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument("--lr_size", default="360,640")
    parser.add_argument("--hr_size", default="1080,1920")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    lr_h, lr_w = (int(x) for x in args.lr_size.split(","))
    hr_h, hr_w = (int(x) for x in args.hr_size.split(","))
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

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

    print(
        f"Benchmark: batch={args.batch_size}, decode={args.decode}, "
        f"lr={lr_h}x{lr_w} hr={hr_h}x{hr_w}"
    )
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("-" * 72)

    for with_model in (False, True):
        mode = "data+train" if with_model else "data-only"
        result = bench_loader(
            f"Codec canvas ({mode})",
            bundle.dataloader,
            device=device,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            with_model=with_model,
            prefetch_batches=args.prefetch_batches,
        )
        print(
            f"{result['name']:<32} "
            f"{result['step_per_s']:6.2f} step/s  "
            f"{result['img_per_s']:7.0f} img/s  "
            f"({result['elapsed_s']:.2f}s / {int(result['steps'])} steps)"
        )
    bundle.close()


if __name__ == "__main__":
    main()
