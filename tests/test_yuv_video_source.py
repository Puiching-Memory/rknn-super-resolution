"""Tests for YUV video source."""

from __future__ import annotations

from pathlib import Path

import torch

from rk3588_mobile_sr.data.types import SourceRecord
from rk3588_mobile_sr.data.yuv_video import (
    YuvVideoSource,
    read_yuv420_frame,
    read_yuv420_patch,
    yuv420_frame_bytes,
)


def _write_gray_yuv(path: Path, width: int, height: int, frames: int) -> None:
    frame_bytes = yuv420_frame_bytes(width, height)
    with path.open("wb") as handle:
        for _ in range(frames):
            y = bytes([128] * (width * height))
            uv = bytes([128] * (width * height // 4))
            handle.write(y + uv + uv)


def test_read_yuv420_frame_shape(tmp_path: Path):
    width, height = 64, 48
    path = tmp_path / "t.yuv"
    _write_gray_yuv(path, width, height, frames=2)
    frame = read_yuv420_frame(path, width=width, height=height, frame_index=1)
    assert frame.shape == (3, height, width)
    assert frame.dtype == torch.float32


def test_read_yuv420_patch_matches_full_crop(tmp_path: Path):
    width, height = 64, 48
    path = tmp_path / "t.yuv"
    _write_gray_yuv(path, width, height, frames=1)
    full = read_yuv420_frame(path, width=width, height=height, frame_index=0)
    patch = read_yuv420_patch(
        path, width=width, height=height, frame_index=0, top=8, left=12, crop_h=32, crop_w=32
    )
    assert patch.shape == (3, 32, 32)
    assert torch.allclose(patch, full[:, 8:40, 12:44], atol=1.5)


def test_yuv_video_source_clip(tmp_path: Path):
    width, height = 64, 48
    path = tmp_path / "t.yuv"
    _write_gray_yuv(path, width, height, frames=5)
    record = SourceRecord.from_dict(
        {
            "id": "uvg/T",
            "type": "yuv_video",
            "path": path.name,
            "width": width,
            "height": height,
            "fps": 30,
            "frames": 5,
            "weight": 1.0,
        }
    )
    source = YuvVideoSource(record, tmp_path)
    clip = source.read_clip(1, 3)
    assert clip.shape == (3, 3, height, width)
