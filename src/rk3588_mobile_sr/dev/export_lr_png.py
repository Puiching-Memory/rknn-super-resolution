"""Export offline LR .npy clips to per-frame PNG (lossless) for DALI image benches."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image


def export_one_npy(npy_path: Path, out_dir: Path) -> tuple[str, int, int]:
    """Write ``stem/00000.png`` ... for one clip. Returns (stem, frames, bytes)."""
    arr = np.load(npy_path)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {arr.shape} for {npy_path}")
    stem = npy_path.stem  # e.g. ..._lr
    clip_dir = out_dir / stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(arr.shape[0]):
        path = clip_dir / f"{i:05d}.png"
        Image.fromarray(arr[i]).save(path, format="PNG", compress_level=3)
        total += path.stat().st_size
    return stem, int(arr.shape[0]), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", type=Path, default=Path("data/raw_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/png_cache"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="max clips (0=all)")
    args = parser.parse_args()

    files = sorted(args.npy_dir.glob("*_lr.npy"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no *_lr.npy under {args.npy_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = 0
    bytes_out = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(export_one_npy, p, args.out_dir) for p in files]
        for i, fut in enumerate(as_completed(futs), 1):
            stem, n, nbytes = fut.result()
            frames += n
            bytes_out += nbytes
            if i % 20 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] last={stem} frames={frames} png={bytes_out/1e9:.2f}GB")

    print(
        f"done: {len(files)} clips, {frames} pngs -> {args.out_dir} "
        f"({bytes_out / (1024**3):.2f} GiB)"
    )


if __name__ == "__main__":
    main()
