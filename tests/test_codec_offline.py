"""Offline codec clip cache and training loader tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from rk3588_mobile_sr.data.codec_build import (
    build_codec_cache_manifest,
    default_codec_workers,
    downscale_clip_to_lr,
    encode_rgb_clip_to_mp4,
)
from rk3588_mobile_sr.data.yuv_video import yuv420_frame_bytes
from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _write_gray_yuv(path: Path, width: int, height: int, frames: int) -> None:
    frame_bytes = yuv420_frame_bytes(width, height)
    with path.open("wb") as handle:
        for _ in range(frames):
            y = bytes([128] * (width * height))
            uv = bytes([128] * (width * height // 4))
            handle.write(y + uv + uv)


def _build_fixture(
    tmp_path: Path,
    *,
    with_mezzanine: bool = True,
    width: int = 96,
    height: int = 72,
    frames: int = 8,
    lr_size: tuple[int, int] = (24, 32),
) -> Path:
    yuv = tmp_path / "clip.yuv"
    _write_gray_yuv(yuv, width, height, frames)
    sources = tmp_path / "sources.jsonl"
    sources.write_text(
        json.dumps(
            {
                "id": "uvg/Test",
                "type": "yuv_video",
                "path": yuv.name,
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

    cache_manifest = build_codec_cache_manifest(
        sources_manifest=sources,
        output_dir=tmp_path / "cache",
        project_root=tmp_path,
        clip_frames=4,
        codecs=("libx264",),
        bitrates_kbps=(400,),
        clips_per_video=1,
        lr_size=lr_size,
        build_hr_mezzanine=with_mezzanine,
        mezzanine_dir=tmp_path / "mezzanine",
        workers=1,
    )
    return cache_manifest


def test_offline_encode_roundtrip_file(tmp_path: Path):
    clip = torch.randint(0, 255, (4, 3, 48, 64), dtype=torch.uint8)
    out = tmp_path / "lr.mp4"
    encode_rgb_clip_to_mp4(clip, out, fps=30, bitrate_kbps=400, gop=4)
    assert out.is_file() and out.stat().st_size > 0
    if shutil.which("ffprobe"):
        from rk3588_mobile_sr.data.codec_build import probe_video_pix_fmt

        assert probe_video_pix_fmt(out) == "yuv420p"


def test_build_codec_cache_manifest(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path, with_mezzanine=True)
    row = json.loads(cache_manifest.read_text().strip())
    assert row["type"] == "codec_clip"
    assert "hr_mp4_path" in row


def test_build_codec_cache_manifest_from_image(tmp_path: Path):
    from torchvision.io import write_png

    image_path = tmp_path / "still.png"
    frame = torch.randint(0, 255, (3, 96, 128), dtype=torch.uint8)
    write_png(frame, str(image_path))
    sources = tmp_path / "sources.jsonl"
    sources.write_text(
        json.dumps(
            {
                "id": "div2k/0001",
                "type": "image",
                "path": image_path.name,
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
    cache_manifest = build_codec_cache_manifest(
        sources_manifest=sources,
        output_dir=tmp_path / "cache",
        project_root=tmp_path,
        clip_frames=24,
        codecs=("libx264", "libx265"),
        bitrates_kbps=(150, 300),
        clips_per_video=2,
        lr_size=(24, 32),
        build_hr_mezzanine=True,
        mezzanine_dir=tmp_path / "mezzanine",
        workers=1,
    )
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
    from rk3588_mobile_sr.data.codec_build import probe_video_pix_fmt

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
    cache_manifest = _build_fixture(tmp_path, lr_size=(24, 32))
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


def test_default_codec_workers_is_positive():
    assert default_codec_workers() >= 1


def test_downscale_clip_to_lr():
    hr = torch.rand(2, 3, 96, 128) * 255.0
    lr = downscale_clip_to_lr(hr, (32, 48))
    assert lr.shape == (2, 3, 32, 48)
