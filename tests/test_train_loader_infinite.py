"""Training loader smoke tests (offline LR .npy + raw YUV)."""

from __future__ import annotations

import pytest

from rk3588_mobile_sr.data.train_loader import resolve_decode_backend


def test_resolve_decode_backend_rejects_torchcodec():
    with pytest.raises(ValueError):
        resolve_decode_backend("torchcodec")


def test_resolve_decode_backend_auto_is_raw():
    assert resolve_decode_backend("auto") == "raw"
    assert resolve_decode_backend("raw") == "raw"


def test_resolve_decode_backend_rejects_dali():
    with pytest.raises(ValueError, match="Unsupported decode"):
        resolve_decode_backend("dali")
