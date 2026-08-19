"""Compare LR storage + read throughput: .npy mmap vs PNG + DALI image decode.

HR path is identical (raw YUV) in both cases; this bench isolates the LR side
that would change if we switched from .npy to PNG+DALI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from rk3588_mobile_sr.data.codec_index import build_codec_frame_index
from rk3588_mobile_sr.data.train_loader import (
    TrainDataSettings,
    apply_canvas_batch_transform,
    build_codec_train_loader,
)
from rk3588_mobile_sr.data.yuv_video import read_yuv420_frame


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_gib(n: int) -> str:
    return f"{n / (1024**3):.2f} GiB"


def build_png_file_list(
    index_entries,
    png_root: Path,
    list_path: Path,
) -> int:
    """Map each LR .npy frame entry to ``png_root/<npy_stem>/<frame:05d>.png``."""
    lines: list[str] = []
    missing = 0
    for i, e in enumerate(index_entries):
        stem = e.lr_path.stem
        png = (png_root / stem / f"{e.lr_frame:05d}.png").resolve()
        if not png.is_file():
            missing += 1
            continue
        # DALI file list: absolute path + label (avoid relative+file_root doubling)
        lines.append(f"{png} {i}")
    if missing:
        raise FileNotFoundError(f"{missing} PNG frames missing under {png_root}")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


@pipeline_def
def _png_lr_pipeline(file_list: str, *, device_id: int):
    # GPU nvJPEG decode when available; falls back inside DALI.
    jpegs, _labels = fn.readers.file(
        file_root="",
        file_list=file_list,
        random_shuffle=False,
        name="lr_reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    return images


class DaliPngLrIterator:
    """LR from DALI PNG decode; HR from raw YUV (same as training)."""

    def __init__(
        self,
        *,
        manifest: Path,
        project_root: Path,
        png_root: Path,
        settings: TrainDataSettings,
        batch_size: int,
        device_id: int,
        seed: int,
        list_path: Path,
        num_threads: int = 4,
    ) -> None:
        self._settings = settings
        self._device = torch.device(f"cuda:{device_id}")
        index = build_codec_frame_index(manifest, project_root=project_root, seed=seed)
        self._entries = index.entries
        n = build_png_file_list(self._entries, png_root, list_path)
        assert n == len(self._entries)

        # Keep HR paths aligned with the (unshuffled-after-list) entry order.
        # file_list order == entries order; DALI reads sequentially with auto_reset.
        self._hr_meta = [
            (e.hr_path, e.hr_width, e.hr_height, e.hr_frame) for e in self._entries
        ]
        self._pos = 0

        pipe = _png_lr_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            file_list=str(list_path),
        )
        pipe.build()
        self._batch_size = batch_size
        self._it = DALIGenericIterator(
            [pipe],
            output_map=["lr"],
            reader_name="lr_reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            batch = next(self._it)
            lr = batch[0]["lr"]  # BHWC uint8 on GPU
            if lr.shape[0] == 0:
                continue
            b = int(lr.shape[0])
            hrs = []
            for _ in range(b):
                path, w, h, fi = self._hr_meta[self._pos % len(self._hr_meta)]
                self._pos += 1
                hrs.append(read_yuv420_frame(path, width=w, height=h, frame_index=fi))
            hr = torch.stack(hrs, dim=0)  # BCHW CPU float
            # lr: BHWC -> BCHW float via transform helper
            if lr.ndim == 4 and lr.shape[-1] == 3:
                pass
            return apply_canvas_batch_transform(
                lr,
                hr,
                lr_size=self._settings.lr_size,
                hr_size=self._settings.hr_size,
                colorspace=self._settings.colorspace,
                nv12_simulate=self._settings.nv12_simulate,
                augment=self._settings.augment,
                device=self._device,
                patch_size=self._settings.patch_size,
                scale=self._settings.scale,
            )

    def close(self) -> None:
        pass


def bench_steps(loader, *, device: torch.device, steps: int, warmup: int) -> dict[str, float]:
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_img = 0
    for _ in range(steps):
        lr, hr = next(it)
        n_img += int(lr.shape[0])
        # touch to ensure materialization
        _ = lr.sum().item(), hr.sum().item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_s": elapsed,
        "step_per_s": steps / elapsed,
        "img_per_s": n_img / elapsed,
        "batch": n_img / steps,
    }


def bench_lr_only_npy(entries, *, batch_size: int, steps: int, warmup: int, device: torch.device):
    """Pure LR .npy mmap read + H2D (no HR, no augment) — upper bound for npy."""
    cache: dict[Path, np.ndarray] = {}
    pos = 0

    def read_batch():
        nonlocal pos
        frames = []
        for _ in range(batch_size):
            e = entries[pos % len(entries)]
            pos += 1
            arr = cache.get(e.lr_path)
            if arr is None:
                arr = np.load(e.lr_path, mmap_mode="r")
                cache[e.lr_path] = arr
            frames.append(np.array(arr[e.lr_frame], copy=True))
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous().float()
        return x.to(device, non_blocking=True)

    for _ in range(warmup):
        read_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        x = read_batch()
        _ = x.sum().item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_s": elapsed,
        "step_per_s": steps / elapsed,
        "img_per_s": steps * batch_size / elapsed,
        "batch": batch_size,
    }


def bench_lr_only_dali_png(
    file_list: Path,
    *,
    batch_size: int,
    steps: int,
    warmup: int,
    device_id: int,
    num_threads: int,
):
    """Pure LR PNG decode via DALI (no HR) — upper bound for PNG+DALI."""

    @pipeline_def
    def pipe():
        jpegs, _ = fn.readers.file(file_list=str(file_list), random_shuffle=False, name="r")
        return fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    p = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    p.build()
    it = DALIGenericIterator(
        [p],
        output_map=["lr"],
        reader_name="r",
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
    )
    for _ in range(warmup):
        next(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_img = 0
    for _ in range(steps):
        batch = next(it)
        lr = batch[0]["lr"]
        n_img += int(lr.shape[0])
        _ = lr.float().sum().item() if hasattr(lr, "float") else float(lr.sum())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_s": elapsed,
        "step_per_s": steps / elapsed,
        "img_per_s": n_img / elapsed,
        "batch": n_img / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/codec_cache/manifest.jsonl")
    parser.add_argument("--npy-dir", type=Path, default=Path("data/raw_cache"))
    parser.add_argument("--png-dir", type=Path, default=Path("data/png_cache"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--dali_threads", type=int, default=4)
    args = parser.parse_args()

    root = Path.cwd()
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

    npy_bytes = _dir_bytes(args.npy_dir)
    png_bytes = _dir_bytes(args.png_dir)
    n_npy = len(list(args.npy_dir.glob("*_lr.npy")))
    n_png = len(list(args.png_dir.rglob("*.png")))

    print("=" * 72)
    print("Disk footprint (LR only)")
    print("-" * 72)
    print(f"  .npy  ({n_npy} clips):  {_fmt_gib(npy_bytes):>10}  ({args.npy_dir})")
    print(f"  PNG   ({n_png} files): {_fmt_gib(png_bytes):>10}  ({args.png_dir})")
    if npy_bytes and png_bytes:
        print(f"  PNG / npy ratio:     {png_bytes / npy_bytes:8.2f}x")
    print()

    settings = TrainDataSettings(
        codec_manifest=args.manifest,
        lr_size=(360, 640),
        hr_size=(1080, 1920),
        colorspace="yuv",
        augment=False,
        decode="raw",
        decode_num_workers=4,
        project_root=str(root),
        patch_size=args.patch_size,
        scale=3,
    )

    index = build_codec_frame_index(args.manifest, project_root=root, seed=0)
    list_path = root / "data" / "_bench_png.list"
    build_png_file_list(index.entries, args.png_dir, list_path)

    print("LR-only decode (+ H2D), no HR / no augment")
    print("-" * 72)
    r_npy = bench_lr_only_npy(
        index.entries,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        device=device,
    )
    print(
        f"  npy mmap          {r_npy['step_per_s']:7.2f} step/s  "
        f"{r_npy['img_per_s']:8.0f} img/s  ({r_npy['elapsed_s']:.2f}s)"
    )
    r_dali = bench_lr_only_dali_png(
        list_path,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        device_id=args.device_id,
        num_threads=args.dali_threads,
    )
    print(
        f"  PNG + DALI        {r_dali['step_per_s']:7.2f} step/s  "
        f"{r_dali['img_per_s']:8.0f} img/s  ({r_dali['elapsed_s']:.2f}s)"
    )
    print()

    print(f"Full train batch (LR+HR YUV, patch={args.patch_size}, no augment)")
    print("-" * 72)
    npy_loader = build_codec_train_loader(
        settings,
        batch_size=args.batch_size,
        seed=0,
        device_id=args.device_id,
    )
    r_full_npy = bench_steps(
        npy_loader.dataloader,
        device=device,
        steps=args.steps,
        warmup=args.warmup,
    )
    print(
        f"  npy+YUV (current) {r_full_npy['step_per_s']:7.2f} step/s  "
        f"{r_full_npy['img_per_s']:8.0f} img/s  ({r_full_npy['elapsed_s']:.2f}s)"
    )
    npy_loader.close()

    dali_it = DaliPngLrIterator(
        manifest=Path(args.manifest),
        project_root=root,
        png_root=args.png_dir,
        settings=settings,
        batch_size=args.batch_size,
        device_id=args.device_id,
        seed=0,
        list_path=list_path,
        num_threads=args.dali_threads,
    )
    r_full_dali = bench_steps(
        dali_it,
        device=device,
        steps=args.steps,
        warmup=args.warmup,
    )
    print(
        f"  PNG DALI+YUV HR   {r_full_dali['step_per_s']:7.2f} step/s  "
        f"{r_full_dali['img_per_s']:8.0f} img/s  ({r_full_dali['elapsed_s']:.2f}s)"
    )
    dali_it.close()
    print("=" * 72)

    summary = {
        "disk": {"npy_bytes": npy_bytes, "png_bytes": png_bytes, "n_npy": n_npy, "n_png": n_png},
        "lr_only": {"npy": r_npy, "dali_png": r_dali},
        "full": {"npy_yuv": r_full_npy, "dali_png_yuv": r_full_dali},
    }
    out = root / "data" / "_bench_npy_vs_png.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
