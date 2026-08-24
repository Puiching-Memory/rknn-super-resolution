"""Tests for VMAF v1 metric helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from rknn_super_resolution.utils.vmaf_metric import (
    batch_vmaf,
    resolve_vmaf_model_arg,
    rgb_chw_to_yuv420_bytes,
)

_REPO = Path(__file__).resolve().parents[1]
_LOCAL_VMAF = _REPO / ".local" / "bin" / "vmaf"
_HAS_VMAF = shutil.which("vmaf") is not None or _LOCAL_VMAF.is_file()


def test_resolve_vmaf_model_aliases():
    assert resolve_vmaf_model_arg("phone") == "version=vmaf_v1.0.16_5d0h"
    assert resolve_vmaf_model_arg("1080p") == "version=vmaf_v1.0.16_3d0h"
    assert resolve_vmaf_model_arg("version=vmaf_v1.0.16_5d0h").startswith("version=")


def test_rgb_to_yuv420_bytes_size():
    rgb = torch.zeros(3, 64, 64)
    raw = rgb_chw_to_yuv420_bytes(rgb)
    assert len(raw) == 64 * 64 * 3 // 2


@pytest.mark.skipif(not _HAS_VMAF, reason="libvmaf/vmaf CLI not installed")
def test_batch_vmaf_identical_near_100():
    from rknn_super_resolution.utils.vmaf_metric import ensure_vmaf_runtime_env

    ensure_vmaf_runtime_env()
    # SpEED in VMAF v1 needs >= ~384px; 384 matches 128 LR patch at 3x.
    x = torch.rand(1, 3, 384, 384) * 255.0
    scores = batch_vmaf(x, x.clone(), model="1080p")
    assert scores.shape == (1,)
    assert float(scores[0]) >= 95.0
