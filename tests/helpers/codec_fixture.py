"""Shared helpers for codec pipeline tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from rk3588_mobile_sr.data.yuv_video import yuv420_frame_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAKEFILE = REPO_ROOT / "scripts/pipeline/Snakefile"
BASE_CONFIG = REPO_ROOT / "scripts/pipeline/config.yaml"


def write_gray_yuv(path: Path, width: int, height: int, frames: int) -> None:
    with path.open("wb") as handle:
        for _ in range(frames):
            y = bytes([128] * (width * height))
            uv = bytes([128] * (width * height // 4))
            handle.write(y + uv + uv)


def write_test_config(
    project_root: Path,
    *,
    clips_per_video: int = 1,
    clip_frames: int = 4,
    codecs: list[str] | None = None,
    bitrates_kbps: list[int] | None = None,
    lr_height: int = 24,
    lr_width: int = 32,
) -> Path:
    """Write a test config.yaml (Snakemake reads this at DAG parse time)."""
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update(
        {
            "clips_per_video": clips_per_video,
            "clip_frames": clip_frames,
            "codecs": codecs or ["libx264"],
            "bitrates_kbps": bitrates_kbps or [400],
            "lr_height": lr_height,
            "lr_width": lr_width,
        }
    )
    out = project_root / "pipeline_test_config.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out


def run_snakemake_pipeline(
    project_root: Path,
    *,
    config_path: Path,
    jobs: int = 4,
    target: str = "write_codec_manifest",
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        "uv",
        "run",
        "snakemake",
        "-j",
        str(jobs),
        "-s",
        str(DEFAULT_SNAKEFILE),
        "--directory",
        str(project_root),
        "--configfile",
        str(config_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["--", target])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"snakemake failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def build_snakemake_codec_fixture(
    tmp_path: Path,
    *,
    with_mezzanine: bool = True,
    width: int = 96,
    height: int = 72,
    frames: int = 8,
    lr_size: tuple[int, int] = (24, 32),
    hr_size: tuple[int, int] | None = None,
) -> Path:
    del with_mezzanine, hr_size
    yuv = tmp_path / "clip.yuv"
    write_gray_yuv(yuv, width, height, frames)
    manifest_dir = tmp_path / "data/sources/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train = manifest_dir / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "id": "uvg/Test",
                "type": "yuv_video",
                "path": str(yuv.relative_to(tmp_path)),
                "width": width,
                "height": height,
                "fps": 30,
                "frames": frames,
                "weight": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = write_test_config(
        tmp_path,
        clips_per_video=1,
        clip_frames=4,
        lr_height=lr_size[0],
        lr_width=lr_size[1],
    )
    run_snakemake_pipeline(tmp_path, config_path=config_path)
    return tmp_path / "data/codec_cache/manifest.jsonl"
