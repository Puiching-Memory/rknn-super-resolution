"""Tests for held-out MLVC checkpoint evaluation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rknn_super_resolution.config import AppConfig
from rknn_super_resolution.eval.mlvc_checkpoint import evaluation_session, load_checkpoint_model
from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.models.graph_format import FLOAT_GRAPH_FORMAT, PT2E_QAT_FORMAT
from rknn_super_resolution.models.qat_utils import prepare_model_for_qat


def _small_config() -> AppConfig:
    config = AppConfig()
    config.model = replace(config.model, num_channels=8, num_blocks=1)
    config.data = replace(config.data, lr_size=(16, 16), hr_size=(48, 48))
    return config


def test_load_checkpoint_model_reconstructs_float_checkpoint(tmp_path: Path):
    config = _small_config()
    source = PhaseRLFNSR(
        num_channels=config.model.num_channels,
        num_blocks=config.model.num_blocks,
        scale=config.model.scale,
        phase_factor=config.model.phase_factor,
    )
    checkpoint = tmp_path / "float.pth"
    torch.save(
        {
            "graph_format": FLOAT_GRAPH_FORMAT,
            "phase": "float",
            "state_dict": source.state_dict(),
        },
        checkpoint,
    )

    restored = load_checkpoint_model(config, checkpoint, torch.device("cpu"))

    assert restored.training is False
    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)


def test_load_checkpoint_model_reconstructs_pt2e_qat_weights(tmp_path: Path):
    config = _small_config()
    source = PhaseRLFNSR(
        num_channels=config.model.num_channels,
        num_blocks=config.model.num_blocks,
    )
    source = prepare_model_for_qat(
        source,
        example_inputs=(
            torch.randn(1, source.core_in_channels, 8, 8),
            torch.randn(1, source.codec_feature_channels, 2, 2),
        ),
    )
    checkpoint = tmp_path / "best_ema.pth"
    torch.save(
        {
            "graph_format": PT2E_QAT_FORMAT,
            "phase": "qat_stable",
            "state_dict": source.state_dict(),
        },
        checkpoint,
    )

    restored = load_checkpoint_model(config, checkpoint, torch.device("cpu"))

    assert restored.training is False
    assert isinstance(restored.core, torch.fx.GraphModule)
    for name, tensor in source.state_dict().items():
        restored_tensor = restored.state_dict()[name]
        if name.endswith("observer_enabled"):
            assert restored_tensor.item() == 0
        else:
            assert torch.equal(restored_tensor, tensor)


def test_load_checkpoint_model_rejects_unversioned_weights(tmp_path: Path):
    checkpoint = tmp_path / "legacy.pth"
    torch.save(PhaseRLFNSR(num_channels=8, num_blocks=1).state_dict(), checkpoint)

    with pytest.raises(TypeError, match="versioned model checkpoint"):
        load_checkpoint_model(_small_config(), checkpoint, torch.device("cpu"))


def test_evaluation_session_supports_direct_single_gpu(monkeypatch: pytest.MonkeyPatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    selected: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    with evaluation_session() as ctx:
        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.device == torch.device("cuda:0")

    assert selected == [torch.device("cuda:0")]


def test_evaluation_session_rejects_partial_torchrun_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(RuntimeError, match="launch with torchrun"), evaluation_session():
        pass
