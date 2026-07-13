"""Combined HR lossless -> LR degradation + codec encode.

Replaces the old ``scale_clip`` + ``lr_encode`` two-step (and the on-disk
``scaled_cache`` rgb24 intermediate). A single ffmpeg filter chain pipes
straight into the LR encoder.

The degradation order models real capture *before* compression::

    HR lossless
      -> optical low-pass blur (HR domain, probabilistic)
      -> sensor noise (HR domain, pre-compression)
      -> mixed downsample kernel (area/bicubic/lanczos, models ISP + optics)
      -> codec encode (x264/x265/av1 @ sampled bitrate/gop)

Post-compression degradations (e.g. JPEG re-compression, light decode noise)
stay in the online augment path (``data/augment.py``) so they are applied in
the correct physical order relative to the codec block structure.
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.codec_args import codec_output_args

# Mixed downsample kernels with empirical weights: area models box-optical,
# bicubic/lanczos model sharper ISP scaling. A single clean area resize does
# not match the optical+ISP blend of real capture.
DOWNSAMPLE_KERNELS: list[tuple[str, float]] = [
    ("area", 0.35),
    ("bicubic", 0.35),
    ("lanczos", 0.30),
]
OPTICAL_BLUR_PROB = 0.5
OPTICAL_BLUR_SIGMA = (0.0, 0.8)  # light optical low-pass; 0 = skip
SENSOR_NOISE_STRENGTH = (2.0, 8.0)  # ffmpeg noise `alls`, 0-100 scale


def job_seed(source_id: str, clip_start: int, codec: str) -> int:
    """Stable per-job seed so a re-run reproduces the same degradation."""
    return abs(hash((source_id, clip_start, codec))) % (2**32)


def build_degrade_vf(
    *,
    lr_w: int,
    lr_h: int,
    rng: random.Random,
) -> str:
    """Build the pre-compression degradation filter chain (HR -> LR)."""
    kernels = [k for k, _ in DOWNSAMPLE_KERNELS]
    weights = [w for _, w in DOWNSAMPLE_KERNELS]
    kernel = rng.choices(kernels, weights=weights, k=1)[0]

    parts: list[str] = []
    # Optical low-pass blur in the HR domain (before subsampling), models the
    # camera anti-aliasing / OLPF filter. Applied probabilistically.
    if rng.random() < OPTICAL_BLUR_PROB:
        sigma = rng.uniform(*OPTICAL_BLUR_SIGMA)
        if sigma > 0.01:
            parts.append(f"gblur=sigma={sigma:.2f}")
    # Sensor noise injected pre-compression (capture-time), temporal so each
    # frame sees independent noise.
    noise = rng.uniform(*SENSOR_NOISE_STRENGTH)
    parts.append(f"noise=alls={noise:.1f}:allf=t")
    # Mixed downsample kernel.
    parts.append(f"scale={lr_w}:{lr_h}:flags={kernel}")
    parts.append("format=yuv420p")
    return ",".join(parts)


def degrade_and_encode(
    hr_path: Path,
    out_path: Path,
    *,
    lr_w: int,
    lr_h: int,
    fps: int,
    frames: int,
    codec: str,
    gop: int,
    bitrate_kbps: int,
    preset: str,
    source_id: str,
    clip_start: int,
) -> None:
    """Run the single ffmpeg degrade+encode for one LR codec clip."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(job_seed(source_id, clip_start, codec))
    vf = build_degrade_vf(lr_w=lr_w, lr_h=lr_h, rng=rng)
    tune = "fastdecode" if codec == "libx264" else None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(hr_path),
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        *codec_output_args(
            codec=codec,
            bitrate_kbps=bitrate_kbps,
            gop=gop,
            preset=preset,
            tune=tune,
        ),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"degrade_and_encode failed for {out_path}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Degraded downsample + LR codec encode (single ffmpeg pass)"
    )
    parser.add_argument("hr_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--lr-w", type=int, required=True)
    parser.add_argument("--lr-h", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--gop", type=int, required=True)
    parser.add_argument("--bitrate", type=int, required=True)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--clip-start", type=int, required=True)
    args = parser.parse_args()
    degrade_and_encode(
        args.hr_path.resolve(),
        args.out_path.resolve(),
        lr_w=args.lr_w,
        lr_h=args.lr_h,
        fps=args.fps,
        frames=args.frames,
        codec=args.codec,
        gop=args.gop,
        bitrate_kbps=args.bitrate,
        preset=args.preset,
        source_id=args.source_id,
        clip_start=args.clip_start,
    )


if __name__ == "__main__":
    main()
