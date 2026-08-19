"""Runtime resolution helpers for canvas codec training."""

from __future__ import annotations

import pytest

from rk3588_mobile_sr.data.train_loader import resolve_decode_backend


def test_resolve_decode_backend_auto():
    assert resolve_decode_backend("auto") == "raw"
    assert resolve_decode_backend("raw") == "raw"


def test_resolve_decode_backend_invalid():
    with pytest.raises(ValueError, match="Unsupported decode"):
        resolve_decode_backend("npu")
    with pytest.raises(ValueError, match="Unsupported decode"):
        resolve_decode_backend("dali")
