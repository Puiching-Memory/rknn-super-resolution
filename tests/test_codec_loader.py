"""Codec raw-frame canvas loader tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rk3588_mobile_sr.data.codec_index import (
    build_codec_frame_index,
    expand_codec_clip_frames,
)
from rk3588_mobile_sr.data.train_loader import (
    TrainDataSettings,
    build_codec_train_loader,
)
from rk3588_mobile_sr.data.types import SourceRecord
from tests.helpers.codec_fixture import build_snakemake_codec_fixture as _build_fixture

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_expand_codec_clip_frames(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path)
    row = json.loads(cache_manifest.read_text().strip().splitlines()[0])
    record = SourceRecord.from_dict(row)
    entries = expand_codec_clip_frames([record], tmp_path)
    assert len(entries) == row["clip_frames"]
    assert entries[0].lr_frame == 0
    # Both LR and HR .npy clips are clip-relative.
    assert entries[0].hr_frame == 0
    assert entries[0].lr_path.suffix == ".npy"
    assert entries[0].hr_path.suffix == ".npy"


def test_build_codec_frame_index(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path)
    index = build_codec_frame_index(
        cache_manifest,
        project_root=tmp_path,
        seed=0,
    )
    assert len(index.entries) >= 1


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_raw_iterator_smoke(tmp_path: Path):
    cache_manifest = _build_fixture(
        tmp_path,
        width=192,
        height=144,
        frames=8,
        lr_size=(64, 96),
    )
    settings = TrainDataSettings(
        codec_manifest=str(cache_manifest),
        lr_size=(64, 96),
        hr_size=(192, 288),
        colorspace="rgb",
        augment=False,
        decode="raw",
        decode_num_workers=0,
        project_root=str(tmp_path),
    )
    bundle = build_codec_train_loader(settings, batch_size=2, seed=0, device_id=0)
    lr, hr = next(bundle.dataloader)
    assert lr.shape[0] == hr.shape[0]
    assert lr.shape[1] == 3 and hr.shape[1] == 3
    bundle.close()


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_raw_iterator_windowed_hr_patch(tmp_path: Path):
    """With patch_size set, HR must come from windowed YUV reads (scale-aligned)."""
    # HR 96x128, LR 32x42 would not be x3; use exact x3 geometry.
    cache_manifest = _build_fixture(
        tmp_path,
        width=192,
        height=144,
        frames=8,
        lr_size=(48, 64),  # 48*3=144, 64*3=192
    )
    settings = TrainDataSettings(
        codec_manifest=str(cache_manifest),
        lr_size=(48, 64),
        hr_size=(144, 192),
        colorspace="rgb",
        augment=False,
        decode="raw",
        decode_num_workers=0,
        project_root=str(tmp_path),
        patch_size=32,
        scale=3,
    )
    bundle = build_codec_train_loader(settings, batch_size=2, seed=0, device_id=0)
    lr, hr = next(bundle.dataloader)
    assert lr.shape == (2, 3, 32, 32)
    assert hr.shape == (2, 3, 96, 96)
    bundle.close()
