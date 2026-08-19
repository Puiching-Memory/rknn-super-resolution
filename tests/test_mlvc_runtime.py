"""Tests for the no-bitstream MLVC runtime contract."""

from __future__ import annotations

from threading import Lock

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
        next_dpb = {"ref_frame": x, "ref_feature": torch.zeros_like(x[:, :1])}
        return {"dpb": next_dpb}


def test_mlvc_small_config_matches_public_checkpoint():
    config = mlvc_model_config("small")
    assert config["type"] == "DMC-6.1sb"
    assert config["feature_channels"] == 48
    assert config["y_channels"] == 48
    assert config["hyperprior_variant"] == "mini"


def test_runtime_pads_360_to_368_and_crops_output():
    runtime = FrozenMLVCRuntime.__new__(FrozenMLVCRuntime)
    runtime.model = _FakeMLVC()
    runtime.device = torch.device("cpu")
    runtime.amp = False
    runtime._lock = Lock()

    sequence = torch.rand(1, 3, 3, 360, 640)
    output = runtime.reconstruct(sequence, torch.tensor([21]))

    assert output.shape == (1, 3, 360, 640)
    assert runtime.model.input_shapes == [(1, 3, 368, 640), (1, 3, 368, 640)]
    assert runtime.model.fa_indices == [1, 0]
    assert torch.allclose(output, sequence[:, -1])
