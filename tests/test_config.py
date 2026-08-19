"""Tests for configuration loading."""

from rk3588_mobile_sr.config import AppConfig, default_config_path, load_config


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.scale == 3
    assert cfg.model.num_channels == 32
    assert cfg.model.negative_slope == 0.1
    assert cfg.data.colorspace == "yuv"
    assert cfg.data.dataset_description.endswith("description.json")
    assert cfg.data.mlvc_variant == "small"
    assert cfg.data.q_indices == (0, 21, 42, 63)
    assert cfg.data.lr_size == (360, 640)
    assert cfg.stage1.patch_size == 128
    assert cfg.deploy.input_h == 360
    assert cfg.deploy.rknn_python == ".venv-rknn/bin/python"
    assert cfg.deploy.rknn_quantize == "kl_divergence"


def test_load_config_from_path():
    path = default_config_path()
    assert path.exists()
    cfg = load_config(path)
    assert cfg.data.num_workers == 4
    assert cfg.data.prefetch_batches == 1
