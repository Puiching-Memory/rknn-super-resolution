"""Tests for RKNN post-conversion accuracy helpers."""

import numpy as np

from rk3588_mobile_sr.deploy.rknn_eval import (
    AccuracyReport,
    AccuracyRow,
    ImagePair,
    _rknn_output_to_hwc,
    collect_image_pairs,
    format_accuracy_table,
    infer_rknn_rgb,
    psnr_numpy,
    rgb_to_nv12_planes,
    ssim_numpy,
)


class _FakeRuntime:
    def inference(self, inputs, data_format=None):
        lr = inputs[0][0].astype(np.float32)
        h, w, _ = lr.shape
        out = np.zeros((3, h * 3, w * 3), dtype=np.float32)
        out[0, :h, :w] = lr[..., 0]
        return [out[None, ...]]


def test_psnr_identical_is_inf():
    img = np.random.rand(8, 8, 3).astype(np.float32) * 255.0
    assert np.isinf(psnr_numpy(img, img))


def test_ssim_identical_is_one():
    img = np.random.rand(16, 16, 3).astype(np.float32) * 255.0
    assert ssim_numpy(img, img) == 1.0


def test_format_accuracy_table_with_fp32():
    report = AccuracyReport(
        num_images=10,
        input_h=360,
        input_w=640,
        fp32=AccuracyRow("FP32 (PyTorch)", 29.25, 0.859, 20.0, 40.0),
        rknn=AccuracyRow("RKNN (INT8)", 28.70, 0.850, 19.0, 39.0),
        match_psnr=35.6,
        match_psnr_min=30.0,
        quant_mode="INT8",
    )
    text = format_accuracy_table(report)
    assert "PSNR vs HR (dB)" in text
    assert "29.250" in text
    assert "28.700" in text
    assert "+0.550" in text
    assert "FP32 vs RKNN output PSNR" in text


def test_infer_rknn_rgb_nhwc_batch():
    lr = np.zeros((360, 640, 3), dtype=np.uint8)
    lr[0, 0] = (10, 20, 30)
    out = infer_rknn_rgb(_FakeRuntime(), lr)
    assert out.shape == (1080, 1920, 3)
    assert out.dtype == np.float32


def test_rknn_output_to_hwc_accepts_nhwc():
    nhwc = np.full((1080, 1920, 3), 128.0, dtype=np.float32)
    out = _rknn_output_to_hwc(nhwc[None, ...])
    assert out.shape == (1080, 1920, 3)


def test_rgb_to_nv12_planes_shape():
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    y, uv = rgb_to_nv12_planes(rgb)
    assert y.shape == (1, 360, 640, 1)
    assert uv.shape == (1, 180, 640, 1)


def test_collect_image_pairs_resizes(tmp_path):
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    hr_dir.mkdir()
    lr_dir.mkdir()

    try:
        import cv2
    except ImportError:
        return

    cv2.imwrite(str(hr_dir / "0001.png"), np.full((30, 30, 3), 200, dtype=np.uint8))
    cv2.imwrite(str(lr_dir / "0001x3.png"), np.full((10, 10, 3), 80, dtype=np.uint8))

    pairs = collect_image_pairs(
        hr_dir,
        lr_dir,
        scale=3,
        input_h=360,
        input_w=640,
        max_images=1,
    )
    assert len(pairs) == 1
    assert pairs[0].lr_rgb.shape == (360, 640, 3)
    assert pairs[0].hr_rgb.shape == (1080, 1920, 3)
