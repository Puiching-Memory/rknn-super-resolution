"""Training loader smoke tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rk3588_mobile_sr.data.train_loader import TrainDataSettings, build_codec_train_loader, resolve_decode_backend
from tests.test_codec_offline import _build_fixture

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_infinite_train_loader(tmp_path: Path):
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
    batches = [next(bundle.dataloader) for _ in range(3)]
    assert all(lr.shape[0] == 2 for lr, _ in batches)
    bundle.close()


def test_resolve_decode_backend_auto():
    backend = resolve_decode_backend("auto")
    assert backend in ("dali", "torchcodec")
