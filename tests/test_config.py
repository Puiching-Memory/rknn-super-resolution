"""Tests for configuration loading."""

from rk3588_mobile_sr.config import AppConfig, default_config_path, load_config


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.scale == 3
    assert cfg.model.num_channels == 32
    assert cfg.stage1.patch_size == 128
    assert cfg.deploy.input_h == 360


def test_load_config_from_path():
    path = default_config_path()
    assert path.exists()
    cfg = load_config(path)
    assert cfg.data.train_hr_dir.endswith("DIV2K_train_HR")
