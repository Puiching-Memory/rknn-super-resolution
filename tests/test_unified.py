"""Tests for plateau-driven FP32→QAT training helpers."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from rknn_super_resolution.distributed.validation import EarlyStopState
from rknn_super_resolution.train.unified import (
    FLOAT,
    QAT_OBSERVE,
    QAT_STABLE,
    _checkpoint,
    _load_raw,
    _stop_reason,
    _validation_config,
    parse_args,
    resolve_training_args,
    validate_training_args,
)
from rknn_super_resolution.utils.train_framework import resolve_model_args


def _parsed_args(monkeypatch, *cli: str):
    monkeypatch.setattr(sys, "argv", ["rknn-super-resolution-train", *cli])
    return resolve_training_args(resolve_model_args(parse_args()))


def test_training_phases_are_the_unified_state_machine():
    assert (FLOAT, QAT_OBSERVE, QAT_STABLE) == ("float", "qat_observe", "qat_stable")


def test_stop_reason_distinguishes_plateau_and_safety_cap():
    plateau = EarlyStopState(enabled=True, patience=1, min_delta=0.1, min_evaluations=1)
    plateau.update(10.0)
    plateau.update(10.0)
    assert _stop_reason(plateau) == "validation plateau"

    cap = EarlyStopState(enabled=True, patience=5, min_delta=0.1, min_evaluations=10)
    cap.update(10.0)
    assert _stop_reason(cap) == "safety step cap"


def test_checkpoint_records_phase_and_step():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    payload = _checkpoint(model, optimizer, step=42, phase=QAT_OBSERVE)
    assert payload["phase"] == QAT_OBSERVE
    assert payload["step"] == 42
    assert "state_dict" in payload
    assert "optimizer" in payload


def test_load_raw_requires_unified_checkpoint(tmp_path):
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(TypeError, match="unified training checkpoint"):
        _load_raw(path, torch.device("cpu"))

    torch.save({"state_dict": {}, "phase": FLOAT, "step": 3}, path)
    with pytest.raises(TypeError, match="unified training checkpoint"):
        _load_raw(path, torch.device("cpu"))

    torch.save(
        {
            "state_dict": {"w": torch.tensor(1.0)},
            "optimizer": {},
            "phase": FLOAT,
            "step": 3,
        },
        path,
    )
    raw = _load_raw(path, torch.device("cpu"))
    assert raw["phase"] == FLOAT
    assert raw["step"] == 3


def test_validate_training_args_rejects_lr_not_aligned_to_phase(monkeypatch):
    args = _parsed_args(monkeypatch)
    args.lr_size = (361, 640)
    with pytest.raises(ValueError, match="phase_factor"):
        validate_training_args(args)


def test_validate_training_args_rejects_short_safety_cap(monkeypatch):
    args = _parsed_args(monkeypatch)
    args.float_safety_max_steps = 1000
    args.val_every = 1000
    args.float_min_evaluations = 12
    with pytest.raises(ValueError, match="float_safety_max_steps"):
        validate_training_args(args)


def test_removed_no_vmaf_flag_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rknn-super-resolution-train", "--no_vmaf"])
    with pytest.raises(SystemExit):
        parse_args()
    args = _parsed_args(monkeypatch)
    assert not hasattr(args, "no_vmaf")


def test_default_val_metric_is_vmaf_for_all_phases(monkeypatch):
    args = _parsed_args(monkeypatch)
    assert args.val_metric == "vmaf"
    assert args.colorspace == "yuv"
    assert args.q_indices == [0, 21, 42, 63]
    assert args.sequence_frames == 8
    float_cfg = _validation_config(args, extended=True, data_preview=True, final_preview=False)
    qat_cfg = _validation_config(args, extended=False, data_preview=False, final_preview=True)
    assert float_cfg.compute_vmaf is True
    assert qat_cfg.compute_vmaf is True
    assert float_cfg.colorspace == "yuv"
    assert qat_cfg.deploy_check is False
    assert qat_cfg.final_preview is True


def test_val_metric_psnr_disables_vmaf_for_all_phases(monkeypatch):
    args = _parsed_args(monkeypatch, "--val_metric", "psnr")
    assert args.val_metric == "psnr"
    float_cfg = _validation_config(args, extended=True, data_preview=True, final_preview=False)
    qat_cfg = _validation_config(args, extended=False, data_preview=False, final_preview=True)
    assert float_cfg.compute_vmaf is False
    assert qat_cfg.compute_vmaf is False


def test_validate_training_args_rejects_unknown_val_metric(monkeypatch):
    args = _parsed_args(monkeypatch)
    args.val_metric = "ssim"
    with pytest.raises(ValueError, match="val_metric"):
        validate_training_args(args)
