"""Decode an mp4 clip into a stacked RGB uint8 .npy (LR or HR).

Offline bake step: ffmpeg decodes once; runtime loaders mmap the .npy and never
touch video codecs. Used for both LR codec clips and lossless HR clips so
train/val share one on-disk format.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def decode_clip_to_npy(
    mp4_path: Path,
    out_npy: Path,
    *,
    width: int,
    height: int,
    frames: int,
) -> None:
    """Decode ``frames`` RGB frames from ``mp4_path`` into ``out_npy``.

    Output shape is ``(frames, height, width, 3)`` uint8.
    """
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"decode_clip_to_npy failed for {mp4_path}: {err}")

    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = frames * height * width * 3
    if arr.size != expected:
        raise RuntimeError(
            f"decoded size {arr.size} != {expected} "
            f"({frames}x{height}x{width}x3) for {mp4_path}"
        )
    clip = np.ascontiguousarray(arr.reshape(frames, height, width, 3))
    np.save(out_npy, clip)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode mp4 clip to stacked RGB uint8 .npy (LR or HR)"
    )
    parser.add_argument("mp4_path", type=Path)
    parser.add_argument("out_npy", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    decode_clip_to_npy(
        args.mp4_path.resolve(),
        args.out_npy.resolve(),
        width=args.width,
        height=args.height,
        frames=args.frames,
    )


if __name__ == "__main__":
    main()
