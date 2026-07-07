"""DALI codec canvas loader tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rk3588_mobile_sr.data.codec_index import (
    build_codec_frame_index,
    expand_codec_clip_frames,
    write_paired_file_lists,
)
from rk3588_mobile_sr.data.types import SourceRecord
from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader, nvidia_cuvid_available
from tests.test_codec_offline import _build_fixture

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_expand_codec_clip_frames(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path)
    row = json.loads(cache_manifest.read_text().strip().splitlines()[0])
    record = SourceRecord.from_dict(row)
    entries = expand_codec_clip_frames([record], tmp_path)
    assert len(entries) == row["clip_frames"]
    assert entries[0].lr_frame == 0
    assert entries[0].hr_frame == row["clip_start"]


def test_write_paired_file_lists_sync(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path)
    row = json.loads(cache_manifest.read_text().strip().splitlines()[0])
    record = SourceRecord.from_dict(row)
    entries = expand_codec_clip_frames([record], tmp_path)
    lr_list = tmp_path / "lr.list"
    hr_list = tmp_path / "hr.list"
    write_paired_file_lists(entries, lr_list_path=lr_list, hr_list_path=hr_list)
    lr_lines = lr_list.read_text().strip().splitlines()
    hr_lines = hr_list.read_text().strip().splitlines()
    assert len(lr_lines) == len(hr_lines) == len(entries)


def test_build_codec_frame_index_for_dali(tmp_path: Path):
    cache_manifest = _build_fixture(tmp_path)
    index = build_codec_frame_index(
        cache_manifest,
        project_root=tmp_path,
        seed=0,
        for_dali=True,
    )
    assert index.lr_list is not None and index.lr_list.is_file()
    assert index.hr_list is not None and index.hr_list.is_file()
    assert len(index.entries) >= 1
    if index.temp_dir is not None:
        index.temp_dir.cleanup()


@pytest.mark.skipif(not nvidia_cuvid_available(), reason="libnvcuvid.so not available")
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_dali_iterator_smoke(tmp_path: Path):
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
        decode="dali",
        project_root=str(tmp_path),
    )
    bundle = build_codec_train_loader(settings, batch_size=2, seed=0, device_id=0)
    lr, hr = next(bundle.dataloader)
    assert lr.shape[0] == hr.shape[0]
    assert lr.shape[1] == 3 and hr.shape[1] == 3
    bundle.close()
