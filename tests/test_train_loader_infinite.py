"""Training loader smoke tests.

The torchcodec CPU fallback has been removed; only DALI (NVDEC) is supported.
Smoke tests that need an actual iterator are gated on CUDA + libnvcuvid.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from rk3588_mobile_sr.data.train_loader import (
    nvidia_cuvid_available,
    resolve_decode_backend,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_resolve_decode_backend_rejects_torchcodec():
    # torchcodec fallback removed -> explicit torchcodec is no longer accepted.
    with pytest.raises(ValueError):
        resolve_decode_backend("torchcodec")


def test_resolve_decode_backend_auto_requires_nvdec():
    if torch.cuda.is_available() and nvidia_cuvid_available():
        assert resolve_decode_backend("auto") == "dali"
    else:
        with pytest.raises(RuntimeError):
            resolve_decode_backend("auto")
