"""Offline codec clip cache and training loader tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from rk3588_mobile_sr.data.codec_index import expand_codec_clip_frames
from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader
from rk3588_mobile_sr.data.types import SourceRecord
import torch.nn.functional as F

from rk3588_mobile_sr.data_pipeline.codec_args import encode_rgb_clip_to_mp4, probe_video_pix_fmt
from tests.helpers.codec_fixture import (
    build_snakemake_codec_fixture,
    run_snakemake_pipeline,
    write_test_config,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_offline_encode_roundtrip_file(tmp_path: Path):
    clip = torch.randint(0, 255, (4, 3, 48, 64), dtype=torch.uint8)
    out = tmp_path / "lr.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, bitrate_kbps=400, gop=4)
    assert out.is_file() and out.stat().st_size > 0
    if shutil.which("ffprobe"):
        assert probe_video_pix_fmt(out) == "yuv420p"


def test_build_codec_cache_manifest(tmp_path: Path):
    cache_manifest = build_snakemake_codec_fixture(tmp_path, with_mezzanine=True)
    row = json.loads(cache_manifest.read_text().strip())
    assert row["type"] == "codec_clip"
    assert "hr_mp4_path" in row


def test_build_codec_cache_manifest_from_image(tmp_path: Path):
    from torchvision.io import write_png

    image_path = tmp_path / "still.png"
    frame = torch.randint(0, 255, (3, 96, 128), dtype=torch.uint8)
    write_png(frame, str(image_path))
    sources = tmp_path / "data/sources/manifests/train.jsonl"
    sources.parent.mkdir(parents=True, exist_ok=True)
    sources.write_text(
        json.dumps(
            {
                "id": "div2k/0001",
                "type": "image",
                "path": str(image_path.relative_to(tmp_path)),
                "width": 96,
                "height": 72,
                "fps": 30,
                "frames": 1,
                "weight": 0.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = write_test_config(
        tmp_path,
        clips_per_video=2,
        clip_frames=24,
        codecs=["libx264", "libx265"],
        bitrates_kbps=[150, 300],
        lr_height=24,
        lr_width=32,
    )
    run_snakemake_pipeline(tmp_path, config_path=config_path)
    cache_manifest = tmp_path / "data/codec_cache/manifest.jsonl"
    rows = [json.loads(line) for line in cache_manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 2 * 2 * 2
    assert rows[0]["source_id"] == "div2k/0001"
    assert rows[0]["clip_frames"] == 1
    assert rows[0]["frames"] == 1
    assert rows[0]["gop"] == 1
    assert rows[0]["encode_mode"] == "intra_only"


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


def test_libsvtav1_intra_single_frame_gop1(tmp_path: Path):
    clip = torch.randint(0, 255, (1, 3, 72, 64), dtype=torch.uint8)
    out = tmp_path / "svt_intra.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, codec="libsvtav1", bitrate_kbps=150, gop=1)
    assert out.is_file() and out.stat().st_size > 0


def test_libsvtav1_low_bitrate_encode(tmp_path: Path):
    clip = torch.randint(0, 255, (1, 3, 72, 64), dtype=torch.uint8)
    out = tmp_path / "svt.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, codec="libsvtav1", bitrate_kbps=150, gop=16)
    assert out.is_file() and out.stat().st_size > 0


def test_torchcodec_train_iterator_smoke(tmp_path: Path):
    cache_manifest = build_snakemake_codec_fixture(
        tmp_path, lr_size=(24, 32), hr_size=(72, 96)
    )
    settings = TrainDataSettings(
        codec_manifest=str(cache_manifest),
        lr_size=(24, 32),
        hr_size=(72, 96),
        colorspace="rgb",
        augment=False,
        decode="torchcodec",
        project_root=str(tmp_path),
    )
    bundle = build_codec_train_loader(settings, batch_size=2, seed=0, device_id=0)
    lr, hr = next(bundle.dataloader)
    assert lr.shape == (2, 3, 24, 32)
    assert hr.shape == (2, 3, 72, 96)
    bundle.close()


def test_area_downscale_shape():
    hr = torch.rand(2, 3, 96, 128) * 255.0
    lr = F.interpolate(hr, size=(32, 48), mode="area")
    assert lr.shape == (2, 3, 32, 48)


def test_div2k_hr_frame_aligns_with_clip_start(tmp_path: Path):
    from torchvision.io import write_png

    image_path = tmp_path / "still.png"
    write_png(torch.randint(0, 255, (3, 128, 160), dtype=torch.uint8), str(image_path))
    train = tmp_path / "data/sources/manifests/train.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text(
        json.dumps(
            {
                "id": "div2k/0001",
                "type": "image",
                "path": str(image_path.relative_to(tmp_path)),
                "width": 96,
                "height": 72,
                "fps": 30,
                "frames": 1,
                "weight": 0.3,
            }
        )
        + "\n",
    )
    config_path = write_test_config(
        tmp_path,
        clips_per_video=4,
        clip_frames=1,
        codecs=["libx264"],
        bitrates_kbps=[400],
        lr_height=24,
        lr_width=32,
    )
    run_snakemake_pipeline(tmp_path, config_path=config_path)
    manifest = tmp_path / "data/codec_cache/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    clip3 = next(row for row in rows if row["clip_start"] == 3)
    record = SourceRecord.from_dict(clip3)
    entries = expand_codec_clip_frames([record], tmp_path)
    assert entries[0].hr_frame == 3
    assert entries[0].lr_frame == 0
