"""FFmpeg codec argument helpers (shared by shell scripts and unit tests)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def libsvtav1_gop(gop: int) -> int:
    """SVT-AV1 VBR cannot force keyframe every frame (gop=1)."""
    return max(gop, 2)


def codec_output_args(
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
        args = [*base, "-preset", preset]
        if tune:
            args += ["-tune", tune]
        return args
    if codec == "libx265":
        return [
            *base,
            "-preset",
            preset,
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
        maxrate_kbps = max(bitrate_kbps + 100, int(bitrate_kbps * 1.5))
        bufsize_kbps = max(maxrate_kbps * 2, maxrate_kbps + 64)
        svt_gop = libsvtav1_gop(gop)
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
) -> None:
    """Encode NCHW RGB uint8/float clip to MP4 via ffmpeg."""
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
        *codec_output_args(
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


def encode_rgb_raw_to_mp4(
    raw_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    codec: str,
    gop: int,
    bitrate_kbps: int,
    preset: str,
) -> None:
    """Encode on-disk rgb24 raw clip to MP4."""
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
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(fps, 1)),
        "-i",
        str(raw_path),
        "-frames:v",
        str(frames),
        *codec_output_args(
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            gop=gop,
            preset=preset,
            tune="fastdecode" if codec == "libx264" else None,
        ),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"lr encode failed for {output_path}: {err}")


def probe_video_pix_fmt(path: Path) -> str:
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
