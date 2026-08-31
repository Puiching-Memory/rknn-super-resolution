"""Tests for the CST Stage A0 synthetic oracle probe."""

import torch

from rknn_super_resolution.dev.cst_oracle import (
    CONDITIONS,
    OracleProbe,
    build_polyphase_observation,
    generate_batch,
    retrieve_global_shifts,
    translate,
)


def test_translate_inverse_restores_integer_shift_interior() -> None:
    image = torch.arange(24 * 24, dtype=torch.float32).reshape(1, 1, 24, 24)
    shift = torch.tensor([[2.0, -1.0]])
    shifted = translate(image, shift)
    restored = translate(shifted, -shift)
    torch.testing.assert_close(restored[..., 3:-3, 3:-3], image[..., 3:-3, 3:-3])


def test_generate_batch_is_repeatable_and_has_expected_shapes() -> None:
    first = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    second = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert first[0].shape == (2, 3, 16, 16)
    assert first[1].shape == (2, 3, 3, 16, 16)
    assert first[2].shape == (2, 3, 2)
    assert first[3].shape == (2, 3, 48, 48)
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected)


def test_all_oracle_conditions_share_one_model_shape() -> None:
    model = OracleProbe(channels=8, blocks=1)
    current, history, shifts, target = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
    )
    for condition in CONDITIONS:
        output = model(current, history, shifts, condition)
        assert output.shape == target.shape
        assert torch.isfinite(output).all()


def test_phase_layout_preserves_output_shape_for_full_and_coarse_context() -> None:
    model = OracleProbe(channels=8, blocks=1, input_layout="phase")
    current, history, shifts, target = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(13),
    )
    for condition in ("spatial", "full", "coarse"):
        output = model(current, history, shifts, condition)
        assert output.shape == target.shape
        assert torch.isfinite(output).all()


def test_hr_layout_preserves_output_shape_for_attention_context() -> None:
    model = OracleProbe(channels=8, blocks=1, input_layout="hr")
    current, history, shifts, target = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(19),
    )
    for condition in ("spatial", "full", "attention", "wrong"):
        output = model(current, history, shifts, condition)
        assert output.shape == target.shape
        assert torch.isfinite(output).all()


def test_polyphase_splat_packs_values_and_observation_counts() -> None:
    current, history, shifts, _target = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(23),
    )
    spatial = build_polyphase_observation(current, history, shifts, "spatial")
    temporal = build_polyphase_observation(current, history, shifts, "full")
    assert spatial.shape == (2, 16, 16, 16)
    assert temporal.shape == spatial.shape
    assert torch.isfinite(temporal).all()
    assert not torch.equal(spatial, temporal)


def test_splat_layout_preserves_output_shape() -> None:
    model = OracleProbe(channels=8, blocks=1, input_layout="splat")
    current, history, shifts, target = generate_batch(
        2,
        48,
        3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(29),
    )
    for condition in ("spatial", "center", "full", "wrong", "retrieved"):
        output = model(current, history, shifts, condition)
        assert output.shape == target.shape
        assert torch.isfinite(output).all()


def test_global_shift_retrieval_recovers_synthetic_motion() -> None:
    current, history, shifts, _target = generate_batch(
        2,
        48,
        2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(41),
    )
    retrieved = retrieve_global_shifts(current, history, crop=2)
    torch.testing.assert_close(retrieved, shifts)
