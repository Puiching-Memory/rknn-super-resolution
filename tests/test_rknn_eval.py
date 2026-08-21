"""Tests for RKNN post-conversion accuracy helpers."""

import numpy as np
import pytest

from rk3588_mobile_sr.deploy.rknn_eval import (
    AccuracyReport,
    AccuracyRow,
    _rknn_output_to_hwc,
    collect_image_pairs,
    format_accuracy_table,
    infer_rknn_rgb,
    mlvc_ycbcr_to_rgb,
    psnr_numpy,
    rgb_to_mlvc_ycbcr,
    ssim_numpy,
)


class _FakePhaseRuntime:
    def inference(self, inputs, data_format=None):
        packed = inputs[0]
        assert packed.shape[1] == 12
        assert data_format == "nchw"
        return [np.zeros((1, 108, packed.shape[2], packed.shape[3]), dtype=np.float32)]


def test_psnr_identical_is_inf():
    img = np.random.rand(8, 8, 3).astype(np.float32) * 255.0
    assert np.isinf(psnr_numpy(img, img))


def test_ssim_identical_is_one():
    img = np.random.rand(16, 16, 3).astype(np.float32) * 255.0
    assert ssim_numpy(img, img) == 1.0


def test_format_accuracy_table_uses_report_scale():
    report = AccuracyReport(
        num_images=10,
        input_h=360,
        input_w=640,
        fp32=AccuracyRow("FP32 (PyTorch)", 29.25, 0.859, 20.0, 40.0),
        rknn=AccuracyRow("RKNN (INT8)", 28.70, 0.850, 19.0, 39.0),
        match_psnr=35.6,
        match_psnr_min=30.0,
        quant_mode="INT8",
        scale=4,
    )
    text = format_accuracy_table(report)
    assert "LR 360x640 -> HR 1440x2560" in text
    assert "PSNR vs HR (dB)" in text
    assert "29.250" in text
    assert "28.700" in text
    assert "+0.550" in text
    assert "FP32 vs RKNN output PSNR" in text


def test_infer_rknn_rgb_phase_core_contract():
    lr = np.zeros((12, 16, 3), dtype=np.uint8)
    out = infer_rknn_rgb(
        _FakePhaseRuntime(),
        lr,
        phase_factor=2,
        scale=3,
    )
    assert out.shape == (36, 48, 3)
    assert out.dtype == np.float32


def test_rknn_output_to_hwc_transposes_nchw_rgb():
    nchw = np.zeros((1, 3, 8, 10), dtype=np.float32)
    nchw[0, 0] = 10.0
    out = _rknn_output_to_hwc(nchw)
    assert out.shape == (8, 10, 3)
    assert out[0, 0, 0] == 10.0


def test_rknn_output_to_hwc_keeps_small_nhwc_rgb():
    nhwc = np.full((8, 10, 3), 128.0, dtype=np.float32)
    out = _rknn_output_to_hwc(nhwc[None, ...])
    assert out.shape == (8, 10, 3)


def test_mlvc_bt709_numpy_roundtrip():
    rgb = np.random.default_rng(0).uniform(0, 255, (16, 24, 3)).astype(np.float32)
    ycbcr = rgb_to_mlvc_ycbcr(rgb)
    restored = mlvc_ycbcr_to_rgb(ycbcr)
    assert ycbcr.shape == rgb.shape
    assert np.allclose(restored, rgb, atol=1e-3)


def test_collect_image_pairs_matches_same_stem(tmp_path):
    cv2 = pytest.importorskip("cv2")
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    hr_dir.mkdir()
    lr_dir.mkdir()

    cv2.imwrite(str(hr_dir / "clip.png"), np.full((30, 40, 3), 200, dtype=np.uint8))
    cv2.imwrite(str(lr_dir / "clip.png"), np.full((10, 10, 3), 80, dtype=np.uint8))

    pairs = collect_image_pairs(
        hr_dir,
        lr_dir,
        scale=3,
        input_h=12,
        input_w=16,
        max_images=1,
    )
    assert len(pairs) == 1
    assert pairs[0].name == "clip.png"
    assert pairs[0].lr_rgb.shape == (12, 16, 3)
    assert pairs[0].hr_rgb.shape == (36, 48, 3)
