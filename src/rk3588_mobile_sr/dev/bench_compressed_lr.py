"""Bench compressed LR caches vs uncompressed .npy: disk + read throughput.

Formats (all lossless, frame-chunked for random access):
  - npy          : current mmap baseline
  - zarr_lz4     : Zarr + Blosc(LZ4), chunks=(1,H,W,3)
  - zarr_zstd    : Zarr + Blosc(ZSTD), chunks=(1,H,W,3)
  - lz4pack      : one file/clip, per-frame LZ4 + offset index

Does not change the training pipeline; experimental only.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lz4.frame
import numpy as np
import torch
import zarr
from numcodecs import Blosc

from rk3588_mobile_sr.data.codec_index import build_codec_frame_index
from rk3588_mobile_sr.data.train_loader import (
    TrainDataSettings,
    apply_canvas_batch_transform,
)
from rk3588_mobile_sr.data.yuv_video import read_yuv420_frame

MAGIC = b"LR4P0001"
HEADER = struct.Struct("<8sIIII")  # magic, n, h, w, reserved


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt_gib(n: int) -> str:
    return f"{n / (1024**3):.3f} GiB"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_zarr(npy_path: Path, out_path: Path, *, cname: str, clevel: int) -> int:
    arr = np.load(npy_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    comp = Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    z = zarr.open(
        str(out_path),
        mode="w",
        shape=arr.shape,
        chunks=(1, arr.shape[1], arr.shape[2], arr.shape[3]),
        dtype=arr.dtype,
        compressor=comp,
    )
    z[:] = arr
    return _dir_bytes(out_path)


def write_lz4pack(npy_path: Path, out_path: Path, *, compression_level: int = 0) -> int:
    arr = np.load(npy_path)
    n, h, w, c = arr.shape
    assert c == 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blobs: list[bytes] = []
    for i in range(n):
        raw = np.ascontiguousarray(arr[i]).tobytes()
        blobs.append(lz4.frame.compress(raw, compression_level=compression_level))
    # header + offset table (n+1 uint64) + payloads
    with out_path.open("wb") as f:
        f.write(HEADER.pack(MAGIC, n, h, w, 0))
        # offsets relative to start of payload region
        table_off = HEADER.size + (n + 1) * 8
        offsets = [0]
        cursor = 0
        for b in blobs:
            cursor += len(b)
            offsets.append(cursor)
        f.write(struct.pack(f"<{n + 1}Q", *offsets))
        assert f.tell() == table_off
        for b in blobs:
            f.write(b)
    return out_path.stat().st_size


def read_lz4pack_frame(path: Path, frame: int) -> np.ndarray:
    with path.open("rb") as f:
        magic, n, h, w, _ = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"bad magic in {path}")
        if not (0 <= frame < n):
            raise IndexError(frame)
        offsets = struct.unpack(f"<{n + 1}Q", f.read((n + 1) * 8))
        payload0 = HEADER.size + (n + 1) * 8
        start, end = offsets[frame], offsets[frame + 1]
        f.seek(payload0 + start)
        blob = f.read(end - start)
    raw = lz4.frame.decompress(blob)
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


def _export_one(args: tuple) -> tuple[str, dict[str, int]]:
    npy_path_s, out_root_s = args
    npy_path = Path(npy_path_s)
    out_root = Path(out_root_s)
    stem = npy_path.stem
    sizes = {
        "zarr_lz4": export_zarr(
            npy_path, out_root / "zarr_lz4" / stem, cname="lz4", clevel=5
        ),
        "zarr_zstd": export_zarr(
            npy_path, out_root / "zarr_zstd" / stem, cname="zstd", clevel=3
        ),
        "lz4pack": write_lz4pack(npy_path, out_root / "lz4pack" / f"{stem}.lr4"),
    }
    return stem, sizes


def export_all(npy_dir: Path, out_root: Path, *, workers: int, limit: int) -> dict[str, int]:
    files = sorted(npy_dir.glob("*_lr.npy"))
    if limit > 0:
        files = files[:limit]
    totals = {"zarr_lz4": 0, "zarr_zstd": 0, "lz4pack": 0}
    jobs = [(str(p), str(out_root)) for p in files]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_export_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, sizes = fut.result()
            for k, v in sizes.items():
                totals[k] += v
            if i % 20 == 0 or i == len(files):
                print(
                    f"export [{i}/{len(files)}] "
                    + " ".join(f"{k}={_fmt_gib(totals[k])}" for k in totals)
                )
    return totals


# ---------------------------------------------------------------------------
# Readers / benches
# ---------------------------------------------------------------------------


class _NpyReader:
    def __init__(self) -> None:
        self._cache: dict[Path, np.ndarray] = {}

    def read(self, path: Path, frame: int) -> np.ndarray:
        arr = self._cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._cache[path] = arr
        return np.array(arr[frame], copy=True)


class _ZarrReader:
    def __init__(self) -> None:
        self._cache: dict[Path, zarr.Array] = {}

    def read(self, path: Path, frame: int) -> np.ndarray:
        z = self._cache.get(path)
        if z is None:
            z = zarr.open(str(path), mode="r")
            self._cache[path] = z
        return np.array(z[frame], copy=True)


class _Lz4PackReader:
    def read(self, path: Path, frame: int) -> np.ndarray:
        return read_lz4pack_frame(path, frame)


def _resolve_paths(entries, *, kind: str, npy_dir: Path, out_root: Path):
    """Map CodecFrameEntry -> (storage_path, frame_index) for each format."""
    mapped = []
    for e in entries:
        stem = e.lr_path.stem
        if kind == "npy":
            mapped.append((e.lr_path, e.lr_frame, e))
        elif kind == "zarr_lz4":
            mapped.append((out_root / "zarr_lz4" / stem, e.lr_frame, e))
        elif kind == "zarr_zstd":
            mapped.append((out_root / "zarr_zstd" / stem, e.lr_frame, e))
        elif kind == "lz4pack":
            mapped.append((out_root / "lz4pack" / f"{stem}.lr4", e.lr_frame, e))
        else:
            raise ValueError(kind)
    return mapped


def bench_lr_only(
    mapped,
    reader,
    *,
    batch_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    pos = 0

    def one_batch():
        nonlocal pos
        frames = []
        for _ in range(batch_size):
            path, fi, _ = mapped[pos % len(mapped)]
            pos += 1
            frames.append(reader.read(path, fi))
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous().float()
        return x.to(device, non_blocking=True)

    for _ in range(warmup):
        one_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        x = one_batch()
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


def bench_full(
    mapped,
    reader,
    *,
    batch_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    settings: TrainDataSettings,
) -> dict[str, float]:
    pos = 0

    def one_batch():
        nonlocal pos
        lrs, hrs = [], []
        for _ in range(batch_size):
            path, fi, e = mapped[pos % len(mapped)]
            pos += 1
            lrs.append(
                torch.from_numpy(reader.read(path, fi)).permute(2, 0, 1).float()
            )
            hrs.append(
                read_yuv420_frame(
                    e.hr_path,
                    width=e.hr_width,
                    height=e.hr_height,
                    frame_index=e.hr_frame,
                )
            )
        lr = torch.stack(lrs)
        hr = torch.stack(hrs)
        return apply_canvas_batch_transform(
            lr,
            hr,
            lr_size=settings.lr_size,
            hr_size=settings.hr_size,
            colorspace=settings.colorspace,
            nv12_simulate=settings.nv12_simulate,
            augment=False,
            device=device,
            patch_size=settings.patch_size,
            scale=settings.scale,
        )

    for _ in range(warmup):
        one_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        lr, hr = one_batch()
        _ = lr.sum().item(), hr.sum().item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_s": elapsed,
        "step_per_s": steps / elapsed,
        "img_per_s": steps * batch_size / elapsed,
        "batch": batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/codec_cache/manifest.jsonl")
    parser.add_argument("--npy-dir", type=Path, default=Path("data/raw_cache"))
    parser.add_argument("--out-root", type=Path, default=Path("data/compr_cache"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=128)
    args = parser.parse_args()

    root = Path.cwd()
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

    if not args.skip_export:
        print("Exporting compressed caches from .npy ...")
        export_all(args.npy_dir, args.out_root, workers=args.workers, limit=args.limit)

    disk = {
        "npy": _dir_bytes(args.npy_dir),
        "zarr_lz4": _dir_bytes(args.out_root / "zarr_lz4"),
        "zarr_zstd": _dir_bytes(args.out_root / "zarr_zstd"),
        "lz4pack": _dir_bytes(args.out_root / "lz4pack"),
        "png": _dir_bytes(Path("data/png_cache")),
    }
    print("=" * 72)
    print("Disk footprint (LR only)")
    print("-" * 72)
    base = disk["npy"] or 1
    for k, v in disk.items():
        print(f"  {k:<10} {_fmt_gib(v):>10}   ({v / base:5.2f}x vs npy)")
    print()

    index = build_codec_frame_index(args.manifest, project_root=root, seed=0)
    settings = TrainDataSettings(
        codec_manifest=args.manifest,
        lr_size=(360, 640),
        hr_size=(1080, 1920),
        colorspace="yuv",
        augment=False,
        decode="raw",
        decode_num_workers=0,
        project_root=str(root),
        patch_size=args.patch_size,
        scale=3,
    )

    readers = {
        "npy": _NpyReader(),
        "zarr_lz4": _ZarrReader(),
        "zarr_zstd": _ZarrReader(),
        "lz4pack": _Lz4PackReader(),
    }

    lr_results = {}
    print("LR-only random read (+ H2D), no HR / no augment")
    print("-" * 72)
    for kind, reader in readers.items():
        mapped = _resolve_paths(
            index.entries, kind=kind, npy_dir=args.npy_dir, out_root=args.out_root
        )
        # sanity
        p0, f0, _ = mapped[0]
        if not (p0.is_file() or p0.is_dir()):
            print(f"  {kind:<10} SKIP (missing {p0})")
            continue
        r = bench_lr_only(
            mapped,
            reader,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
        )
        lr_results[kind] = r
        print(
            f"  {kind:<10} {r['step_per_s']:7.2f} step/s  "
            f"{r['img_per_s']:8.0f} img/s  ({r['elapsed_s']:.2f}s)"
        )
    print()

    full_results = {}
    print(f"Full train batch (LR + HR YUV, patch={args.patch_size}, no augment)")
    print("-" * 72)
    for kind, reader in readers.items():
        mapped = _resolve_paths(
            index.entries, kind=kind, npy_dir=args.npy_dir, out_root=args.out_root
        )
        p0, _, _ = mapped[0]
        if not (p0.is_file() or p0.is_dir()):
            continue
        r = bench_full(
            mapped,
            reader,
            batch_size=args.batch_size,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            settings=settings,
        )
        full_results[kind] = r
        print(
            f"  {kind:<10} {r['step_per_s']:7.2f} step/s  "
            f"{r['img_per_s']:8.0f} img/s  ({r['elapsed_s']:.2f}s)"
        )
    print("=" * 72)

    out = {
        "disk": disk,
        "lr_only": lr_results,
        "full": full_results,
    }
    out_path = root / "data" / "_bench_compr.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
