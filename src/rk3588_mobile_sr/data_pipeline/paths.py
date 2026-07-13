"""Path helpers shared by the Snakemake pipeline."""

from __future__ import annotations

from pathlib import Path


def safe_source_id(source_id: str) -> str:
    return source_id.replace("/", "__")


def hr_clip_path(hr_dir: Path, *, source_id: str, clip_start: int) -> Path:
    """Path to the per-clip lossless HR mp4 (one clip per file)."""
    return hr_dir / f"{safe_source_id(source_id)}_s{clip_start}_hr.mp4"


def lr_mp4_path(
    cache_dir: Path,
    *,
    source_id: str,
    clip_start: int,
    codec: str,
    bitrate_kbps: int,
    gop: int,
) -> Path:
    name = f"{safe_source_id(source_id)}_s{clip_start}_g{gop}_{codec}_{bitrate_kbps}k.mp4"
    return cache_dir / name
