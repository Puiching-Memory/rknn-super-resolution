"""Tests for frozen-MLVC batch processing."""

from __future__ import annotations

from typing import Any

import torch

from rk3588_mobile_sr.data.mlvc_loader import (
    MLVCBatchProcessor,
    MLVCDeviceBatch,
    mlvc_ycbcr_to_rgb,
    rgb_to_mlvc_ycbcr,
)


class _LastFrameRuntime:
    def __init__(self) -> None:
        self.q_index: torch.Tensor | None = None

    def reconstruct(self, sequence: torch.Tensor, q_index: torch.Tensor) -> torch.Tensor:
        self.q_index = q_index.clone()
        return sequence[:, -1]


class _TensorBatchDecoder:
    def decode_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["lr_sequence"], batch["hr"]

    def close(self) -> None:
        return None


def test_mlvc_colorspace_roundtrip():
    rgb = torch.rand(2, 3, 12, 20)
    restored = mlvc_ycbcr_to_rgb(rgb_to_mlvc_ycbcr(rgb))
    assert torch.allclose(restored, rgb, atol=1e-5)


def test_validation_processor_uses_fixed_q_and_canvas_shapes():
    runtime = _LastFrameRuntime()
    processor = MLVCBatchProcessor(
        runtime,
        decoder=_TensorBatchDecoder(),
        device=torch.device("cpu"),
        q_indices=(0, 21, 42, 63),
        colorspace="yuv",
        patch_size=None,
        scale=3,
    )
    batch = {
        "lr_sequence": torch.randint(0, 256, (2, 3, 3, 12, 20), dtype=torch.uint8),
        "hr": torch.randint(0, 256, (2, 3, 36, 60), dtype=torch.uint8),
        "q_index": torch.tensor([21, 63]),
    }
    lr, hr = processor(batch, training=False)
    assert lr.shape == (2, 3, 12, 20)
    assert hr.shape == (2, 3, 36, 60)
    assert runtime.q_index is not None
    assert runtime.q_index.tolist() == [21, 63]
    assert lr.is_contiguous() and hr.is_contiguous()


def test_training_processor_crops_scale_aligned_patches():
    processor = MLVCBatchProcessor(
        _LastFrameRuntime(),
        decoder=_TensorBatchDecoder(),
        device=torch.device("cpu"),
        q_indices=(21,),
        colorspace="rgb",
        patch_size=8,
        scale=3,
    )
    batch = {
        "lr_sequence": torch.randint(0, 256, (1, 2, 3, 12, 20), dtype=torch.uint8),
        "hr": torch.randint(0, 256, (1, 3, 36, 60), dtype=torch.uint8),
    }
    lr, hr = processor(batch, training=True)
    assert lr.shape == (1, 3, 8, 8)
    assert hr.shape == (1, 3, 24, 24)


def test_device_batch_preserves_tuple_contract_on_cpu():
    lr = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 24, 24)
    batch = MLVCDeviceBatch(lr, hr)
    batch.wait_ready()
    unpacked_lr, unpacked_hr = batch
    assert unpacked_lr is lr
    assert unpacked_hr is hr
