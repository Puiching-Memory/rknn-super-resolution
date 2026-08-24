"""Tests for the no-bitstream MLVC runtime contract."""

from __future__ import annotations

from threading import Lock

import pytest
import torch
import torch.nn as nn

from rk3588_mobile_sr.data.mlvc_runtime import FrozenMLVCRuntime, mlvc_model_config


class _FakeMLVC(nn.Module):
    padding_size = 16
    frame_index_map = (0, 1, 0, 2, 0, 2, 0, 2)

    def __init__(self) -> None:
        super().__init__()
        self.input_shapes: list[tuple[int, ...]] = []
        self.fa_indices: list[int] = []

    def compress_core(self, x, dpb, q_index, fa_idx):
        assert dpb["ref_frame"].shape == x.shape
        assert q_index.shape == (x.shape[0],)
        self.input_shapes.append(tuple(x.shape))
        self.fa_indices.append(fa_idx)
        feature = torch.zeros(x.shape[0], 96, x.shape[-2] // 8, x.shape[-1] // 8)
        next_dpb = {"ref_frame": x, "ref_feature": feature}
        return {"dpb": next_dpb}


def test_mlvc_small_config_matches_public_checkpoint():
    assert mlvc_model_config("small") == {
        "type": "DMC-6.1sb",
        "activation": "LeakyReLU",
        "input_offset": -0.5,
        "memory_activation": "identity",
        "zero_init_residual": True,
        "chunk_mode": "gated",
        "ffn_gate_activation": "ReLU1",
        "chain_feature_adaptors": True,
        "feature_channels": 48,
        "spatial_prior_channels": 128,
        "recon_channels": 192,
        "hidden_channels": 192,
        "hyperprior_num_blocks": 2,
        "y_scale_repeat": 4,
        "z_channels": 48,
        "y_channels": 48,
        "hyperprior_variant": "mini",
        "feature_extractor_num_conv1_layers": 1,
        "feature_extractor_num_conv2_layers": 1,
    }


def test_mlvc_model_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unsupported MLVC variant"):
        mlvc_model_config("tiny")


def test_runtime_pads_360_to_368_and_crops_output():
    runtime = FrozenMLVCRuntime.__new__(FrozenMLVCRuntime)
    runtime.model = _FakeMLVC()
    runtime.device = torch.device("cpu")
    runtime.amp = False
    runtime._lock = Lock()

    sequence = torch.rand(1, 3, 3, 360, 640)
    output = runtime.reconstruct(sequence, torch.tensor([21]))

    assert output.frames.shape == (1, 2, 3, 360, 640)
    assert output.features.shape == (1, 2, 96, 46, 80)
    assert runtime.model.input_shapes == [(1, 3, 368, 640), (1, 3, 368, 640)]
    assert runtime.model.fa_indices == [1, 0]
    assert torch.allclose(output.frames[:, -1], sequence[:, -1])
