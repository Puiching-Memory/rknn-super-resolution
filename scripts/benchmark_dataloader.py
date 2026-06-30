"""Benchmark data loading and training step throughput."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from rk3588_mobile_sr.data.div2k_dali import build_dali_train_loader
from rk3588_mobile_sr.data.div2k_loader import build_dataloader
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
) -> dict[str, float]:
    model = None
    criterion = None
    optimizer = None
    if with_model:
        model = MobileOneSR().to(device)
        model.train()
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_steps(data_iter, n: int):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        count = 0
        while count < n:
            try:
                lr, hr = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                lr, hr = next(data_iter)
            if lr.device.type != "cuda" or lr.device.index != device.index:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
            if with_model:
                optimizer.zero_grad(set_to_none=True)
                out = model(lr)
                loss = criterion(out, hr)
                loss.backward()
                optimizer.step()
            count += 1
        torch.cuda.synchronize()
        return time.perf_counter() - t0, data_iter

    data_iter = iter(loader)
    for _ in range(warmup):
        _, data_iter = run_steps(data_iter, 1)

    elapsed, _ = run_steps(data_iter, steps)
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
    parser.add_argument("--hr_dir", default="data/DIV2K_train_HR")
    parser.add_argument("--lr_dir", default="data/DIV2K_train_LR_bicubic/X3")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

    print(f"Benchmark: batch={args.batch_size}, patch={args.patch_size}, steps={args.steps}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print("-" * 72)

    results: list[dict[str, float]] = []

    for with_model in (False, True):
        mode = "data+train" if with_model else "data-only"
        pytorch_loader, _ = build_dataloader(
            args.hr_dir,
            args.lr_dir,
            scale=3,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=4,
            augment=True,
        )
        results.append(
            bench_loader(
                f"PyTorch DataLoader ({mode})",
                pytorch_loader,
                device=device,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                with_model=with_model,
            )
        )
        dali_loader = build_dali_train_loader(
            args.hr_dir,
            args.lr_dir,
            scale=3,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device_id=args.device_id,
            shard_id=0,
            num_shards=1,
            num_threads=4,
            augment=True,
        )
        results.append(
            bench_loader(
                f"DALI ({mode})",
                dali_loader,
                device=device,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                with_model=with_model,
            )
        )

    for r in results:
        print(
            f"{r['name']:<32} "
            f"{r['step_per_s']:6.2f} step/s  "
            f"{r['img_per_s']:7.0f} img/s  "
            f"({r['elapsed_s']:.2f}s / {int(r['steps'])} steps)"
        )

    pytorch_train = next(r for r in results if r["name"] == "PyTorch DataLoader (data+train)")
    dali_train = next(r for r in results if r["name"] == "DALI (data+train)")
    speedup = dali_train["step_per_s"] / pytorch_train["step_per_s"]
    print("-" * 72)
    print(f"DALI vs PyTorch (data+train): {speedup:.2f}x step/s")


if __name__ == "__main__":
    main()
