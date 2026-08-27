"""Tests for plateau-driven validation state machine (EarlyStopState)."""

from __future__ import annotations

from rknn_super_resolution.train.validation import (
    EarlyStopState,
    ValidationConfig,
    primary_metric_logs,
)


def test_early_stop_improves_resets_patience():
    state = EarlyStopState(enabled=True, patience=3, min_delta=0.1)
    improved, stop = state.update(25.0)
    assert improved and not stop
    assert state.best_score == 25.0
    assert state.patience_counter == 0

    improved, stop = state.update(25.5)
    assert improved and not stop
    assert state.patience_counter == 0


def test_primary_metric_logs_keeps_vmaf_out_of_best_psnr():
    logs = primary_metric_logs(
        primary_key="val/vmaf",
        best_score=41.53,
        psnr_at_best=31.35,
    )
    assert logs["val/best_score"] == 41.53
    assert logs["val/best_vmaf"] == 41.53
    assert logs["val/best_psnr"] == 31.35


def test_early_stop_triggers_after_patience():
    state = EarlyStopState(enabled=True, patience=2, min_delta=0.1)
    state.update(20.0)

    _, stop = state.update(20.05)
    assert not stop
    assert state.patience_counter == 1

    _, stop = state.update(20.05)
    assert stop
    assert state.patience_counter == 2


def test_early_stop_honors_minimum_evaluations():
    state = EarlyStopState(
        enabled=True,
        patience=1,
        min_delta=0.1,
        min_evaluations=4,
    )
    state.update(20.0)
    assert state.update(20.0)[1] is False
    assert state.update(20.0)[1] is False
    assert state.update(20.0)[1] is True
    assert state.evaluations == 4


def test_early_stop_disabled_never_stops():
    state = EarlyStopState(enabled=False, patience=1, min_delta=0.1)
    for psnr in (10.0, 9.0, 8.0):
        _, stop = state.update(psnr)
        assert not stop


def test_validation_config_defaults_to_yuv():
    assert ValidationConfig().colorspace == "yuv"
