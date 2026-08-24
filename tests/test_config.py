"""Tests for configuration loading and deploy contracts."""

import pytest

from rknn_super_resolution.config import AppConfig, DeployConfig, default_config_path, load_config


def test_default_config_deploy_contract():
    cfg = load_config()
    assert cfg.model.scale == 3
    assert cfg.model.num_channels == 32
    assert cfg.model.num_blocks == 4
    assert cfg.model.phase_factor == 2
    assert cfg.model.codec_feature_channels == 96
    assert cfg.model.codec_project_channels == 16
    assert cfg.model.negative_slope == 0.1
    assert cfg.data.colorspace == "yuv"
    assert cfg.data.mlvc_variant == "small"
    assert cfg.data.q_indices == (0, 21, 42, 63)
    assert cfg.data.lr_size == (360, 640)
    assert cfg.data.hr_size == (1080, 1920)
    assert cfg.data.codec_context is True
    assert cfg.training.val_metric == "vmaf"
    assert cfg.deploy.target == "rk3576"
    assert cfg.deploy.input_h == 360
    assert cfg.deploy.input_w == 640


def test_load_config_from_bundled_yaml():
    path = default_config_path()
    assert path.exists()
    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert cfg.data.dataset_description.endswith("frame_sequences.csv")
    assert cfg.model.in_channels == 3
    assert cfg.model.out_channels == 3


@pytest.mark.parametrize("target", ["rk3576", "rk3588", "RK3588", "rv1126b", "RV1126B"])
def test_deploy_config_accepts_tested_boards(target: str):
    assert DeployConfig(target=target).target == target.lower()


def test_deploy_config_does_not_restrict_toolkit_targets():
    assert DeployConfig(target="RK9999").target == "rk9999"


def test_deploy_config_rejects_empty_target():
    with pytest.raises(ValueError, match="cannot be empty"):
        DeployConfig(target="  ")


def test_load_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("deploy:\n  target: rk3576\n  old_target: rk3588\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown DeployConfig field.*old_target"):
        load_config(path)


def test_load_config_rejects_unknown_sections(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("legacy_deploy: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown config section.*legacy_deploy"):
        load_config(path)
