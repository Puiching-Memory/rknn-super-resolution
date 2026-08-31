"""Tests for the direct CST information-ceiling oracle."""

import torch

from rknn_super_resolution.dev.cst_information_oracle import (
    evaluate_shift_retrieval,
    predict_observations,
    reconstruct,
    save_reconstruction_panel,
)
from rknn_super_resolution.dev.cst_oracle import generate_batch


def test_predict_observations_has_expected_shapes() -> None:
    estimate = torch.rand(2, 3, 48, 48)
    shifts = torch.tensor(
        [
            [[-0.5, 1.0], [1.5, -1.0]],
            [[1.0, 0.5], [-1.5, -0.5]],
        ]
    )
    current, history = predict_observations(estimate, shifts)
    assert current.shape == (2, 3, 16, 16)
    assert history.shape == (2, 2, 3, 16, 16)


def test_shift_retrieval_returns_bounded_metrics() -> None:
    current, history, shifts, _target = generate_batch(
        2,
        48,
        2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(31),
    )
    metrics = evaluate_shift_retrieval(current, history, shifts, crop=2)
    assert 0.0 <= metrics["exact_accuracy"] <= 1.0
    assert 0.0 <= metrics["top3_accuracy"] <= 1.0
    assert metrics["offset_mae"] >= 0.0


def test_reconstruct_returns_finite_metrics_on_cpu() -> None:
    current, history, shifts, target = generate_batch(
        1,
        48,
        2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(17),
    )
    reconstruction, metrics = reconstruct(
        current,
        history,
        shifts,
        target,
        use_history=True,
        steps=2,
        learning_rate=0.01,
        tv_weight=2e-4,
        crop=3,
    )
    assert reconstruction.shape == target.shape
    assert torch.isfinite(reconstruction).all()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_save_reconstruction_panel(tmp_path) -> None:
    target = torch.rand(1, 3, 48, 48)
    current = torch.rand(1, 3, 16, 16)
    path = tmp_path / "panel.png"
    save_reconstruction_panel(
        path,
        target=target,
        current=current,
        spatial=target * 0.9,
        temporal=target * 0.95,
    )
    assert path.is_file()
