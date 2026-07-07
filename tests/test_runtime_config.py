"""Runtime resolution helpers for canvas codec training."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from rk3588_mobile_sr.data.train_loader import resolve_decode_backend


def test_resolve_decode_backend_auto():
    with patch.object(torch.cuda, "is_available", return_value=True):
        with patch(
            "rk3588_mobile_sr.data.train_loader.nvidia_cuvid_available",
            return_value=True,
        ):
            assert resolve_decode_backend("auto") == "dali"
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert resolve_decode_backend("auto") == "torchcodec"


def test_resolve_decode_backend_dali_requires_cuvid():
    with patch(
        "rk3588_mobile_sr.data.train_loader.nvidia_cuvid_available",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="decode=dali requires"):
            resolve_decode_backend("dali")


def test_resolve_decode_backend_invalid():
    with pytest.raises(ValueError, match="Unsupported decode"):
        resolve_decode_backend("npu")
