"""Build offline codec LR clip cache + JSONL manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rk3588_mobile_sr.data.image_source import StillImageSource
from rk3588_mobile_sr.data.manifest import load_manifest
from rk3588_mobile_sr.data.types import SourceRecord
from rk3588_mobile_sr.data.yuv_video import YuvVideoSource

DEFAULT_CODECS: tuple[str, ...] = ("libx264", "libx265", "libsvtav1")
DEFAULT_BITRATES_KBPS: tuple[int, ...] = (150, 200, 300, 500, 800)


def default_codec_workers() -> int:
    """Default ffmpeg parallelism for offline codec cache builds."""
    return min(32, os.cpu_count() or 4)


def downscale_clip_to_lr(
    hr_clip: torch.Tensor,
    lr_size: tuple[int, int],
) -> torch.Tensor:
    """Resize HR RGB clip NCHW float [0,255] to LR canvas."""
    lr_h, lr_w = lr_size
    if hr_clip.ndim == 3:
        x = hr_clip.unsqueeze(0)
        y = F.interpolate(x, size=(lr_h, lr_w), mode="area")
        return y.squeeze(0)
    y = F.interpolate(hr_clip, size=(lr_h, lr_w), mode="area")
    return y


def _libsvtav1_gop(gop: int) -> int:
    """SVT-AV1 VBR cannot force keyframe every frame (gop=1); stills stay intra-only."""
    return max(gop, 2)


def _codec_output_args(
    *,
    codec: str,
    bitrate_kbps: int,
    gop: int,
    preset: str,
    tune: str | None,
) -> list[str]:
    bufsize = f"{max(bitrate_kbps * 2, bitrate_kbps + 64)}k"
    base = [
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-bf",
        "0",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        bufsize,
    ]
    if codec == "libx264":
        base += ["-preset", preset]
        if tune:
            base += ["-tune", tune]
        return base
    if codec == "libx265":
        return [
            *base,
            "-preset",
            preset,
            # hvc1 is required for broad MP4 player compatibility (QuickTime/Safari/etc.).
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:repeat-headers=1",
        ]
    if codec == "libsvtav1":
        svt_preset = {
            "ultrafast": 12,
            "superfast": 11,
            "veryfast": 10,
            "faster": 9,
            "fast": 8,
            "medium": 7,
        }.get(preset, 10)
        # SVT-AV1 VBR requires maxrate > target (CBR is not supported in RA mode).
        maxrate_kbps = max(bitrate_kbps + 100, int(bitrate_kbps * 1.5))
        bufsize_kbps = max(maxrate_kbps * 2, maxrate_kbps + 64)
        svt_gop = _libsvtav1_gop(gop)
        return [
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(svt_gop),
            "-bf",
            "0",
            "-b:v",
            f"{bitrate_kbps}k",
            "-maxrate",
            f"{maxrate_kbps}k",
            "-bufsize",
            f"{bufsize_kbps}k",
            "-preset",
            str(svt_preset),
            "-svtav1-params",
            "log-level=error",
        ]
    raise ValueError(f"Unsupported codec {codec!r}")


def encode_rgb_clip_to_mp4(
    clip: torch.Tensor,
    output_path: Path,
    *,
    fps: int,
    codec: str = "libx264",
    gop: int = 16,
    bitrate_kbps: int = 500,
    preset: str = "veryfast",
    tune: str | None = "fastdecode",
    output_pix_fmt: str = "yuv420p",
) -> None:
    """Encode NCHW RGB uint8/float clip to MP4 via ffmpeg (offline batch job)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clip.ndim == 3:
        clip = clip.unsqueeze(0)
    _, _, h, w = clip.shape
    raw = clip.byte().clamp(0, 255).permute(0, 2, 3, 1).contiguous().cpu().numpy().tobytes()
    tune_arg = tune if codec == "libx264" else None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(max(fps, 1)),
        "-i",
        "pipe:0",
        *_codec_output_args(
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            gop=gop,
            preset=preset,
            tune=tune_arg,
        ),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"offline encode failed for {output_path}: {err}")


def encode_yuv420_file_to_mp4(
    yuv_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    codec: str = "libx264",
    gop: int = 1,
    crf: int = 18,
    preset: str = "veryfast",
    output_pix_fmt: str = "yuv420p",
) -> None:
    """Encode a raw YUV420p file to a high-quality mezzanine MP4 (offline)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(fps, 1)),
        "-i",
        str(yuv_path),
        "-c:v",
        codec,
        "-pix_fmt",
        output_pix_fmt,
        "-g",
        str(gop),
        "-preset",
        preset,
        "-crf",
        str(crf),
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"mezzanine encode failed for {output_path}: {err}")


def probe_video_pix_fmt(path: Path) -> str:
    """Return video stream pix_fmt via ffprobe (e.g. ``yuv420p``)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt",
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        raise RuntimeError(f"ffprobe failed for {path}: {err}")
    return proc.stdout.strip()


def _lr_mp4_relpath(
    cache_dir: Path,
    *,
    source_id: str,
    clip_start: int,
    codec: str,
    bitrate_kbps: int,
    gop: int,
) -> Path:
    safe_id = source_id.replace("/", "__")
    name = f"{safe_id}_s{clip_start}_g{gop}_{codec}_{bitrate_kbps}k.mp4"
    return cache_dir / name


def _hr_mezzanine_relpath(mezzanine_dir: Path, source_id: str) -> Path:
    safe_id = source_id.replace("/", "__")
    return mezzanine_dir / f"{safe_id}_hr.mp4"


def _ensure_yuv_mezzanine(
    record: SourceRecord,
    root: Path,
    mezzanine_dir: Path,
    *,
    skip_existing: bool,
    mezzanine_crf: int,
    mezzanine_gop: int,
    encode_pix_fmt: str,
) -> Path:
    mezzanine_dir.mkdir(parents=True, exist_ok=True)
    out_path = _hr_mezzanine_relpath(mezzanine_dir, record.id).resolve()
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return out_path.relative_to(root)
    yuv_path = (root / record.path).resolve()
    encode_yuv420_file_to_mp4(
        yuv_path,
        out_path,
        width=record.width,
        height=record.height,
        fps=record.fps or 30,
        gop=mezzanine_gop,
        crf=mezzanine_crf,
        output_pix_fmt=encode_pix_fmt,
    )
    return out_path.relative_to(root)


def _ensure_image_mezzanine(
    record: SourceRecord,
    root: Path,
    mezzanine_dir: Path,
    *,
    skip_existing: bool,
    mezzanine_frames: int,
    mezzanine_gop: int,
    mezzanine_crf: int,
    encode_pix_fmt: str,
) -> Path:
    mezzanine_dir.mkdir(parents=True, exist_ok=True)
    out_path = _hr_mezzanine_relpath(mezzanine_dir, record.id).resolve()
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return out_path.relative_to(root)
    source = StillImageSource(record, root)
    del mezzanine_frames  # still images are always encoded as a single frame
    hr_clip = source.read_clip(0)
    encode_rgb_clip_to_mp4(
        hr_clip,
        out_path,
        fps=source.fps,
        codec="libx264",
        gop=mezzanine_gop,
        bitrate_kbps=20_000,
        preset="veryfast",
        tune=None,
        output_pix_fmt=encode_pix_fmt,
    )
    del mezzanine_crf  # image mezzanine uses a high-quality intermediate encode
    return out_path.relative_to(root)


def _clip_starts_for_record(
    record: SourceRecord,
    *,
    clip_frames: int,
    clips_per_video: int,
    rng: random.Random,
) -> list[int]:
    """Deterministic clip start indices without reading source pixels."""
    if record.type == "image":
        return list(range(max(1, clips_per_video)))
    max_start = max(0, record.frames - clip_frames)
    if max_start == 0:
        return [0]
    return sorted(
        {rng.randint(0, max_start) for _ in range(min(clips_per_video, max_start + 1))}
    )


@dataclass(frozen=True)
class MezzanineTask:
    record: SourceRecord
    root: str
    mezzanine_dir: str
    skip_existing: bool
    mezzanine_crf: int
    mezzanine_gop: int
    mezzanine_frames: int
    encode_pix_fmt: str


@dataclass(frozen=True)
class LrEncodeTask:
    output_path: str
    fps: int
    codec: str
    gop: int
    bitrate_kbps: int
    preset: str
    encode_pix_fmt: str
    skip_existing: bool


@dataclass(frozen=True)
class ClipEncodeBatch:
    """One HR clip downscaled once, then encoded to many LR variants."""

    clip_np: np.ndarray
    tasks: tuple[LrEncodeTask, ...]


def _run_mezzanine_task(task: MezzanineTask) -> tuple[str, str]:
    root = Path(task.root)
    mezz_root = Path(task.mezzanine_dir)
    record = task.record
    if record.type == "yuv_video":
        hr_rel = _ensure_yuv_mezzanine(
            record,
            root,
            mezz_root,
            skip_existing=task.skip_existing,
            mezzanine_crf=task.mezzanine_crf,
            mezzanine_gop=task.mezzanine_gop,
            encode_pix_fmt=task.encode_pix_fmt,
        )
    else:
        hr_rel = _ensure_image_mezzanine(
            record,
            root,
            mezz_root,
            skip_existing=task.skip_existing,
            mezzanine_frames=task.mezzanine_frames,
            mezzanine_gop=task.mezzanine_gop,
            mezzanine_crf=task.mezzanine_crf,
            encode_pix_fmt=task.encode_pix_fmt,
        )
    return record.id, hr_rel.as_posix()


def _run_clip_encode_batch(batch: ClipEncodeBatch) -> None:
    clip = torch.from_numpy(batch.clip_np)
    for task in batch.tasks:
        out_path = Path(task.output_path)
        if task.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            continue
        encode_rgb_clip_to_mp4(
            clip,
            out_path,
            fps=task.fps,
            codec=task.codec,
            gop=task.gop,
            bitrate_kbps=task.bitrate_kbps,
            preset=task.preset,
            output_pix_fmt=task.encode_pix_fmt,
        )


def _lr_output_ready(lr_path: Path, *, skip_existing: bool) -> bool:
    return skip_existing and lr_path.is_file() and lr_path.stat().st_size > 0


def _manifest_row_for_clip(
    *,
    record: SourceRecord,
    root: Path,
    cache_dir: Path,
    clip_start: int,
    clip_frames: int,
    codec: str,
    bitrate_kbps: int,
    gop: int,
    encode_mode: str,
    hr_paths: dict[str, str],
    fps: int,
    lr_size: tuple[int, int],
) -> dict:
    lr_rel = _lr_mp4_relpath(
        cache_dir,
        source_id=record.id,
        clip_start=clip_start,
        codec=codec,
        bitrate_kbps=bitrate_kbps,
        gop=gop,
    )
    lr_path = lr_rel.resolve()
    row_id = f"{record.id}@s{clip_start}@{codec}@{bitrate_kbps}k"
    row: dict = {
        "id": row_id,
        "type": "codec_clip",
        "path": lr_path.relative_to(root).as_posix(),
        "weight": record.weight,
        "width": record.width,
        "height": record.height,
        "fps": fps,
        "frames": clip_frames,
        "source_id": record.id,
        "source_path": record.path,
        "clip_start": clip_start,
        "clip_frames": clip_frames,
        "codec": codec,
        "bitrate_kbps": bitrate_kbps,
        "gop": gop,
        "lr_height": lr_size[0],
        "lr_width": lr_size[1],
        "encode_mode": encode_mode,
        "tags": ["codec_offline"],
    }
    if record.id in hr_paths:
        row["hr_mp4_path"] = hr_paths[record.id]
    return row


def _plan_clip_manifest_and_tasks(
    *,
    record: SourceRecord,
    root: Path,
    cache_dir: Path,
    clip_start: int,
    clip_frames: int,
    codecs: tuple[str, ...],
    bitrates_kbps: tuple[int, ...],
    gop: int,
    preset: str,
    encode_pix_fmt: str,
    encode_mode: str,
    hr_paths: dict[str, str],
    skip_existing: bool,
    fps: int,
    lr_size: tuple[int, int],
) -> tuple[list[dict], list[LrEncodeTask]]:
    """Plan manifest rows and encode tasks without reading HR pixels."""
    rows: list[dict] = []
    tasks: list[LrEncodeTask] = []
    for codec in codecs:
        for bitrate in bitrates_kbps:
            lr_rel = _lr_mp4_relpath(
                cache_dir,
                source_id=record.id,
                clip_start=clip_start,
                codec=codec,
                bitrate_kbps=bitrate,
                gop=gop,
            )
            lr_path = lr_rel.resolve()
            rows.append(
                _manifest_row_for_clip(
                    record=record,
                    root=root,
                    cache_dir=cache_dir,
                    clip_start=clip_start,
                    clip_frames=clip_frames,
                    codec=codec,
                    bitrate_kbps=bitrate,
                    gop=gop,
                    encode_mode=encode_mode,
                    hr_paths=hr_paths,
                    fps=fps,
                    lr_size=lr_size,
                )
            )
            if not _lr_output_ready(lr_path, skip_existing=skip_existing):
                tasks.append(
                    LrEncodeTask(
                        output_path=str(lr_path),
                        fps=fps,
                        codec=codec,
                        gop=gop,
                        bitrate_kbps=bitrate,
                        preset=preset,
                        encode_pix_fmt=encode_pix_fmt,
                        skip_existing=skip_existing,
                    )
                )
    return rows, tasks


def _plan_codec_rows(
    *,
    record: SourceRecord,
    root: Path,
    cache_dir: Path,
    clip_start: int,
    hr_clip: torch.Tensor,
    lr_size: tuple[int, int],
    clip_frames: int,
    codecs: tuple[str, ...],
    bitrates_kbps: tuple[int, ...],
    gop: int,
    preset: str,
    encode_pix_fmt: str,
    encode_mode: str,
    hr_paths: dict[str, str],
    skip_existing: bool,
    fps: int,
) -> tuple[list[dict], ClipEncodeBatch | None]:
    rows, tasks = _plan_clip_manifest_and_tasks(
        record=record,
        root=root,
        cache_dir=cache_dir,
        clip_start=clip_start,
        clip_frames=clip_frames,
        codecs=codecs,
        bitrates_kbps=bitrates_kbps,
        gop=gop,
        preset=preset,
        encode_pix_fmt=encode_pix_fmt,
        encode_mode=encode_mode,
        hr_paths=hr_paths,
        skip_existing=skip_existing,
        fps=fps,
        lr_size=lr_size,
    )
    if not tasks:
        return rows, None
    lr_clip = downscale_clip_to_lr(hr_clip, lr_size)
    clip_np = lr_clip.byte().clamp(0, 255).cpu().numpy()
    return rows, ClipEncodeBatch(clip_np=clip_np, tasks=tuple(tasks))


def _drain_futures(pending: set) -> None:
    for fut in wait(pending, return_when=FIRST_COMPLETED).done:
        fut.result()


def _run_thread_pool_collect(tasks, worker_fn, *, workers: int) -> list:
    task_list = list(tasks)
    if not task_list:
        return []
    if workers <= 1:
        return [worker_fn(task) for task in task_list]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_fn, task) for task in task_list]
        return [fut.result() for fut in futures]


def _run_thread_pool(
    tasks,
    worker_fn,
    *,
    workers: int,
    max_pending: int | None = None,
) -> None:
    """Run ffmpeg-friendly tasks on a bounded thread pool (no pickling)."""
    task_iter = iter(tasks)
    if workers <= 1:
        for task in task_iter:
            worker_fn(task)
        return

    pending_cap = max_pending or workers * 4
    pending: set = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in task_iter:
            pending.add(pool.submit(worker_fn, task))
            if len(pending) >= pending_cap:
                _drain_futures(pending)
        while pending:
            _drain_futures(pending)


def build_codec_cache_manifest(
    *,
    sources_manifest: str | Path,
    output_dir: str | Path,
    manifest_out: str | Path | None = None,
    project_root: str | Path | None = None,
    clip_frames: int = 24,
    gop: int = 16,
    codecs: tuple[str, ...] = DEFAULT_CODECS,
    bitrates_kbps: tuple[int, ...] = DEFAULT_BITRATES_KBPS,
    clips_per_video: int = 8,
    lr_size: tuple[int, int] = (360, 640),
    preset: str = "veryfast",
    seed: int = 42,
    skip_existing: bool = True,
    build_hr_mezzanine: bool = True,
    mezzanine_dir: str | Path = "data/mezzanine",
    encode_pix_fmt: str = "yuv420p",
    mezzanine_crf: int = 18,
    mezzanine_gop: int = 1,
    mezzanine_frames: int = 24,
    workers: int | None = None,
) -> Path:
    """Pre-encode temporal-GOP LR clips and write ``codec_clip`` manifest rows."""
    root = Path(project_root or Path.cwd()).resolve()
    cache_dir = (root / output_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    mezz_root = (root / mezzanine_dir).resolve()
    out_path = Path(manifest_out or cache_dir / "manifest.jsonl").resolve()
    parallel_workers = default_codec_workers() if workers is None else max(1, workers)

    records = load_manifest(sources_manifest, project_root=root)
    train_sources = [r for r in records if r.type in ("yuv_video", "image")]
    if not train_sources:
        raise ValueError("No yuv_video/image sources in manifest")

    rng = random.Random(seed)
    rows: list[dict] = []
    hr_paths: dict[str, str] = {}

    if build_hr_mezzanine:
        mezz_tasks = [
            MezzanineTask(
                record=record,
                root=str(root),
                mezzanine_dir=str(mezz_root),
                skip_existing=skip_existing,
                mezzanine_crf=mezzanine_crf,
                mezzanine_gop=mezzanine_gop,
                mezzanine_frames=mezzanine_frames,
                encode_pix_fmt=encode_pix_fmt,
            )
            for record in train_sources
            if not (
                skip_existing
                and _hr_mezzanine_relpath(mezz_root, record.id)
                .resolve()
                .is_file()
                and _hr_mezzanine_relpath(mezz_root, record.id).resolve().stat().st_size > 0
            )
        ]
        if mezz_tasks:
            for record_id, hr_rel in _run_thread_pool_collect(
                mezz_tasks, _run_mezzanine_task, workers=parallel_workers
            ):
                hr_paths[record_id] = hr_rel
        for record in train_sources:
            hr_rel = _hr_mezzanine_relpath(mezz_root, record.id).resolve()
            if hr_rel.is_file() and hr_rel.stat().st_size > 0:
                hr_paths[record.id] = hr_rel.relative_to(root).as_posix()

    encode_batches = 0
    encode_jobs = 0
    pending_cap = parallel_workers * 4

    def _iter_clip_batches():
        nonlocal encode_batches, encode_jobs
        yuv_sources: dict[str, YuvVideoSource] = {}
        image_sources: dict[str, StillImageSource] = {}
        for record in train_sources:
            if record.type == "yuv_video":
                starts = _clip_starts_for_record(
                    record,
                    clip_frames=clip_frames,
                    clips_per_video=clips_per_video,
                    rng=rng,
                )
                for clip_start in starts:
                    clip_rows, tasks = _plan_clip_manifest_and_tasks(
                        record=record,
                        root=root,
                        cache_dir=cache_dir,
                        clip_start=clip_start,
                        clip_frames=clip_frames,
                        codecs=codecs,
                        bitrates_kbps=bitrates_kbps,
                        gop=gop,
                        preset=preset,
                        encode_pix_fmt=encode_pix_fmt,
                        encode_mode="temporal_gop",
                        hr_paths=hr_paths,
                        skip_existing=skip_existing,
                        fps=record.fps or 30,
                        lr_size=lr_size,
                    )
                    rows.extend(clip_rows)
                    if not tasks:
                        continue
                    if record.id not in yuv_sources:
                        yuv_sources[record.id] = YuvVideoSource(record, root)
                    hr_clip = yuv_sources[record.id].read_clip(clip_start, clip_frames)
                    lr_clip = downscale_clip_to_lr(hr_clip, lr_size)
                    clip_np = lr_clip.byte().clamp(0, 255).cpu().numpy()
                    batch = ClipEncodeBatch(clip_np=clip_np, tasks=tuple(tasks))
                    encode_batches += 1
                    encode_jobs += len(tasks)
                    yield batch
            elif record.type == "image":
                starts = _clip_starts_for_record(
                    record,
                    clip_frames=1,
                    clips_per_video=clips_per_video,
                    rng=rng,
                )
                for clip_start in starts:
                    clip_rows, tasks = _plan_clip_manifest_and_tasks(
                        record=record,
                        root=root,
                        cache_dir=cache_dir,
                        clip_start=clip_start,
                        clip_frames=1,
                        codecs=codecs,
                        bitrates_kbps=bitrates_kbps,
                        gop=1,
                        preset=preset,
                        encode_pix_fmt=encode_pix_fmt,
                        encode_mode="intra_only",
                        hr_paths=hr_paths,
                        skip_existing=skip_existing,
                        fps=record.fps or 30,
                        lr_size=lr_size,
                    )
                    rows.extend(clip_rows)
                    if not tasks:
                        continue
                    if record.id not in image_sources:
                        image_sources[record.id] = StillImageSource(record, root)
                    hr_clip = image_sources[record.id].read_clip(clip_start)
                    lr_clip = downscale_clip_to_lr(hr_clip, lr_size)
                    clip_np = lr_clip.byte().clamp(0, 255).cpu().numpy()
                    batch = ClipEncodeBatch(clip_np=clip_np, tasks=tuple(tasks))
                    encode_batches += 1
                    encode_jobs += len(tasks)
                    yield batch

    print(
        f"encoding clip batches with {parallel_workers} workers "
        f"(streamed planning + ffmpeg)"
    )
    _run_thread_pool(
        _iter_clip_batches(),
        _run_clip_encode_batch,
        workers=parallel_workers,
        max_pending=pending_cap,
    )
    print(f"encoded {encode_batches} clip batches ({encode_jobs} mp4 files)")

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} codec_clip entries -> {out_path}")
    if build_hr_mezzanine:
        print(f"hr mezzanine: {len(hr_paths)} sources under {mezz_root}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline codec LR clip cache")
    parser.add_argument(
        "--sources",
        default="data/sources/manifests/train.jsonl",
        help="source manifest with yuv_video/image rows",
    )
    parser.add_argument("--output-dir", default="data/codec_cache")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--mezzanine-dir", default="data/mezzanine")
    parser.add_argument("--no-hr-mezzanine", action="store_true")
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--gop", type=int, default=16)
    parser.add_argument("--codecs", default=",".join(DEFAULT_CODECS))
    parser.add_argument("--bitrates", default=",".join(str(b) for b in DEFAULT_BITRATES_KBPS))
    parser.add_argument("--clips-per-video", type=int, default=8)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--encode-pix-fmt", default="yuv420p")
    parser.add_argument("--mezzanine-crf", type=int, default=18)
    parser.add_argument("--mezzanine-gop", type=int, default=1)
    parser.add_argument("--mezzanine-frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="parallel ffmpeg workers")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    build_codec_cache_manifest(
        sources_manifest=args.sources,
        output_dir=args.output_dir,
        manifest_out=args.manifest_out,
        clip_frames=args.clip_frames,
        gop=args.gop,
        codecs=tuple(c.strip() for c in args.codecs.split(",") if c.strip()),
        bitrates_kbps=tuple(int(x) for x in args.bitrates.split(",") if x.strip()),
        clips_per_video=args.clips_per_video,
        preset=args.preset,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
        build_hr_mezzanine=not args.no_hr_mezzanine,
        mezzanine_dir=args.mezzanine_dir,
        encode_pix_fmt=args.encode_pix_fmt,
        mezzanine_crf=args.mezzanine_crf,
        mezzanine_gop=args.mezzanine_gop,
        mezzanine_frames=args.mezzanine_frames,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
