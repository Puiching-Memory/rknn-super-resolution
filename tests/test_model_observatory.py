"""Tests for bounded model observation statistics and panels."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from rknn_super_resolution.dev.model_observatory import (
    _activation_report,
    _checkpoint_report,
    load_checkpoint_tensors,
)
from rknn_super_resolution.utils.model_observatory import (
    ForwardHookObservatory,
    TensorObservatory,
    analyze_state_dict,
    observe_tensor,
    render_tensor_observatory,
    summarize_tensor,
    summarize_transition,
)


def test_tensor_summary_exposes_clip_error_and_exact_rank() -> None:
    tensor = torch.diag(torch.tensor([2.0, 2.0]))
    summary = summarize_tensor(
        "identity",
        tensor,
        clip_abs=1.0,
        channel_dim=0,
        include_structure=True,
    )
    assert summary.clip_ratio == 0.5
    assert summary.clip_relative_rmse == pytest.approx(0.5)
    assert summary.stable_rank == pytest.approx(2.0)
    assert summary.effective_rank == pytest.approx(2.0)


def test_observation_is_bounded_and_keeps_visual_sketches() -> None:
    tensor = torch.arange(2 * 4 * 18 * 24, dtype=torch.float32).reshape(2, 4, 18, 24)
    observation = observe_tensor("body", tensor, map_size=(48, 32), max_samples=128)
    assert observation.histogram_density.shape == (64,)
    assert observation.channel_rms is not None
    assert observation.channel_rms.shape == (4,)
    assert observation.spatial_map is not None
    assert observation.spatial_map.shape == (32, 48)
    assert observation.spatial_map.min() >= 0.0
    assert observation.spatial_map.max() <= 1.0


def test_transition_reports_state_growth_and_update_direction() -> None:
    previous = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    current = previous * 2.0
    summary = summarize_transition("state", previous, current)
    assert summary.growth_ratio == pytest.approx(2.0)
    assert summary.relative_delta_rms == pytest.approx(1.0)
    assert summary.cosine_similarity == pytest.approx(1.0)
    assert summary.sign_flip_ratio == 0.0


def test_state_dict_analysis_includes_spectra_and_frequency() -> None:
    state = {
        "conv.weight": torch.randn(8, 4, 3, 3),
        "conv.bias": torch.zeros(8),
    }
    observations, spectra, frequencies = analyze_state_dict(state)
    assert len(observations) == 2
    assert spectra["conv.weight"].shape == (8,)
    assert frequencies["conv.weight"].shape == (32, 32)


def test_tensor_observatory_renders_activation_and_transition_panel() -> None:
    observatory = TensorObservatory(clip_abs=1.0, max_samples=128)
    previous = torch.randn(1, 4, 12, 16)
    current = previous + 0.1
    observatory.observe("feature", current)
    observatory.observe_transition("feature", previous, current)
    panel = render_tensor_observatory(observatory)
    assert panel.dtype == np.uint8
    assert panel.ndim == 3
    assert panel.shape[1] == 1580
    assert "observatory/feature/rms" in observatory.scalar_metrics()
    assert "observatory/feature/growth_ratio" in observatory.scalar_metrics()


def test_forward_hook_observatory_collects_leaf_module_outputs() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 2, 1))
    with ForwardHookObservatory(model) as hooks:
        model(torch.randn(1, 3, 8, 8))
    assert set(hooks.observatory.observations) == {"0", "2"}


def test_checkpoint_report_accepts_versioned_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "small.pth"
    torch.save(
        {
            "phase": "float",
            "step": 12,
            "state_dict": {
                "conv.weight": torch.randn(4, 3, 3, 3),
                "conv.bias": torch.zeros(4),
            },
        },
        checkpoint,
    )
    key, tensors, metadata = load_checkpoint_tensors(checkpoint)
    assert key == "state_dict"
    assert set(tensors) == {"conv.weight", "conv.bias"}
    assert metadata["step"] == 12

    output = tmp_path / "report"
    paths = _checkpoint_report(
        checkpoint,
        output,
        state_key=None,
        clip_abs=1.0,
        max_layers=8,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_activation_report_pairs_matching_temporal_arrays(tmp_path: Path) -> None:
    previous = tmp_path / "previous.npz"
    current = tmp_path / "current.npz"
    np.savez(previous, state=np.ones((1, 4, 6, 8), dtype=np.float32))
    np.savez(current, state=np.full((1, 4, 6, 8), 2.0, dtype=np.float32))
    paths = _activation_report(current, previous, tmp_path / "report", clip_abs=4.0)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    payload = paths["activation_summary"].read_text(encoding="utf-8")
    assert '"growth_ratio": 2.0' in payload
