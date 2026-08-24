"""Tests for frozen-MLVC batch processing."""

from __future__ import annotations

from typing import Any

import torch

from rknn_super_resolution.data.mlvc_loader import (
    MLVCBatchProcessor,
    MLVCDeviceBatch,
    mlvc_ycbcr_to_rgb,
    rgb_to_mlvc_ycbcr,
)
from rknn_super_resolution.data.mlvc_runtime import MLVCReconstruction


class _PassthroughRuntime:
    def __init__(self) -> None:
        self.q_index: torch.Tensor | None = None
        self.input_shape: tuple[int, ...] | None = None

    def reconstruct(self, sequence: torch.Tensor, q_index: torch.Tensor) -> MLVCReconstruction:
        self.q_index = q_index.clone()
        self.input_shape = tuple(sequence.shape)
        batch, time = sequence.shape[:2]
        features = torch.ones(batch, time - 1, 96, 2, 3)
        return MLVCReconstruction(sequence[:, 1:], features)


class _TensorBatchDecoder:
    def decode_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["lr_sequence"], batch["hr"]

    def close(self) -> None:
        return None


def test_mlvc_colorspace_roundtrip():
    rgb = torch.rand(2, 3, 12, 20)
    restored = mlvc_ycbcr_to_rgb(rgb_to_mlvc_ycbcr(rgb))
    assert torch.allclose(restored, rgb, atol=1e-5)


def test_validation_processor_uses_fixed_q_and_last_p_frame_canvas():
    runtime = _PassthroughRuntime()
    processor = MLVCBatchProcessor(
        runtime,
        decoder=_TensorBatchDecoder(),
        device=torch.device("cpu"),
        q_indices=(0, 21, 42, 63),
        colorspace="yuv",
        scale=3,
    )
    batch = {
        "lr_sequence": torch.randint(0, 256, (2, 3, 3, 12, 20), dtype=torch.uint8),
        "hr": torch.randint(0, 256, (2, 3, 3, 36, 60), dtype=torch.uint8),
        "q_index": torch.tensor([21, 63]),
    }
    lr, hr = processor(batch, training=False)
    assert runtime.input_shape == (2, 3, 3, 12, 20)
    assert lr.shape == (2, 3, 12, 20)
    assert hr.shape == (2, 3, 36, 60)
    assert runtime.q_index is not None
    assert runtime.q_index.tolist() == [21, 63]
    assert lr.is_contiguous() and hr.is_contiguous()


def test_training_processor_flattens_all_p_frames_on_full_canvas():
    runtime = _PassthroughRuntime()
    processor = MLVCBatchProcessor(
        runtime,
        decoder=_TensorBatchDecoder(),
        device=torch.device("cpu"),
        q_indices=(21,),
        colorspace="rgb",
        scale=3,
    )
    batch = {
        "lr_sequence": torch.randint(0, 256, (1, 4, 3, 12, 20), dtype=torch.uint8),
        "hr": torch.randint(0, 256, (1, 4, 3, 36, 60), dtype=torch.uint8),
    }
    lr, hr = processor(batch, training=True)
    assert runtime.input_shape == (1, 4, 3, 12, 20)
    assert lr.shape == (3, 3, 12, 20)
    assert hr.shape == (3, 3, 36, 60)


def test_device_batch_preserves_tuple_contract_on_cpu():
    lr = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 24, 24)
    batch = MLVCDeviceBatch(lr, hr)
    batch.wait_ready()
    unpacked_lr, unpacked_hr = batch
    assert unpacked_lr is lr
    assert unpacked_hr is hr


def test_codec_context_is_optional_and_flattens_with_frames():
    processor = MLVCBatchProcessor(
        _PassthroughRuntime(),
        decoder=_TensorBatchDecoder(),
        device=torch.device("cpu"),
        q_indices=(21,),
        colorspace="yuv",
        scale=3,
        codec_context=True,
        codec_dropout=0.0,
    )
    batch = {
        "lr_sequence": torch.randint(0, 256, (1, 4, 3, 12, 20), dtype=torch.uint8),
        "hr": torch.randint(0, 256, (1, 4, 3, 36, 60), dtype=torch.uint8),
    }
    (current, codec), target = processor(batch, training=True)
    assert current.shape == (3, 3, 12, 20)
    assert codec.shape == (3, 96, 2, 3)
    assert target.shape == (3, 3, 36, 60)
