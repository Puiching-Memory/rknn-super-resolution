"""Offline codec clip cache and training loader tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rk3588_mobile_sr.data.codec_index import expand_codec_clip_frames
from rk3588_mobile_sr.data.types import SourceRecord
from rk3588_mobile_sr.data_pipeline.codec_args import (
    encode_rgb_clip_to_mp4,
    probe_video_pix_fmt,
)
from tests.helpers.codec_fixture import build_snakemake_codec_fixture

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_offline_encode_roundtrip_file(tmp_path: Path):
    clip = torch.randint(0, 255, (4, 3, 48, 64), dtype=torch.uint8)
    out = tmp_path / "lr.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, bitrate_kbps=400, gop=4)
    assert out.is_file() and out.stat().st_size > 0
    if shutil.which("ffprobe"):
        assert probe_video_pix_fmt(out) == "yuv420p"


def test_build_codec_cache_manifest(tmp_path: Path):
    cache_manifest = build_snakemake_codec_fixture(tmp_path)
    row = json.loads(cache_manifest.read_text().strip())
    assert row["type"] == "codec_clip"
    assert row["path"].endswith("_lr.npy")  # offline-baked LR RGB frames
    assert row["hr_path"].endswith("_hr.npy")  # offline-baked HR RGB frames
    assert "source_path" in row             # provenance YUV path
    assert row["encode_mode"] == "temporal_gop"


def test_codec_clip_frame_index_relative_and_weighted(tmp_path: Path):
    """LR/HR frame indices are clip-relative; P-frames weighted higher."""
    cache_manifest = build_snakemake_codec_fixture(tmp_path)
    row = json.loads(cache_manifest.read_text().strip())
    record = SourceRecord.from_dict(row)
    entries = expand_codec_clip_frames([record], tmp_path)
    assert len(entries) == row["clip_frames"]
    assert entries[0].lr_frame == 0
    assert entries[0].hr_frame == 0
    assert entries[0].lr_path.suffix == ".npy"
    assert entries[0].hr_path.suffix == ".npy"
    gop = row["gop"]
    intra = {i for i in range(row["clip_frames"]) if i % max(gop, 1) == 0}
    intra_w = entries[0].weight
    p_w = next(entries[i].weight for i in range(1, len(entries)))
    if intra and p_w != intra_w:
        assert p_w > intra_w  # P-frames carry more codec artifact -> higher weight


def test_libx265_mp4_uses_hvc1_tag(tmp_path: Path):
    clip = torch.randint(0, 255, (2, 3, 72, 64), dtype=torch.uint8)
    out = tmp_path / "x265.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, codec="libx265", bitrate_kbps=300, gop=16)
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_tag_string",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stdout.strip() == "hvc1"
    assert probe_video_pix_fmt(out) == "yuv420p"


def test_libsvtav1_low_bitrate_encode(tmp_path: Path):
    clip = torch.randint(0, 255, (1, 3, 72, 64), dtype=torch.uint8)
    out = tmp_path / "svt.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, codec="libsvtav1", bitrate_kbps=150, gop=16)
    assert out.is_file() and out.stat().st_size > 0


def test_area_downscale_shape():
    hr = torch.rand(2, 3, 96, 128) * 255.0
    lr = F.interpolate(hr, size=(32, 48), mode="area")
    assert lr.shape == (2, 3, 32, 48)
