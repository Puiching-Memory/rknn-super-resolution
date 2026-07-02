"""YAML configuration loading for MobileOneSR."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    scale: int = 3
    in_channels: int = 3
    out_channels: int = 3
    num_channels: int = 32
    num_blocks: int = 8
    num_conv_branches: int = 4


@dataclass
class DataConfig:
    train_hr_dir: str = "data/DIV2K_train_HR"
    train_lr_dir: str = "data/DIV2K_train_LR_bicubic/X3"
    val_hr_dir: str = "data/DIV2K_valid_HR"
    val_lr_dir: str = "data/DIV2K_valid_LR_bicubic/X3"
    num_workers: int = 4


@dataclass
class Stage1Config:
    patch_size: int = 128
    batch_size: int = 16
    max_steps: int = 100_000
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.01
    val_every: int = 1000
    lr: float = 1e-3
    loss: str = "l1"


@dataclass
class Stage2Config:
    patch_size: int = 160
    batch_size: int = 16
    max_steps: int = 80_000
    val_every: int = 4000
    early_stop_patience: int = 8
    early_stop_min_delta: float = 0.005
    lr: float = 3e-5
    lambda_dct: float = 0.02
    lambda_kd: float = 0.03
    teacher_arch: str = "mambairv2_light"
    teacher_weight: str = "checkpoints/teacher/mambairv2_lightSR_x3.pth"


@dataclass
class Stage3QatConfig:
    patch_size: int = 144
    batch_size: int = 1
    max_steps: int = 15_000
    phase1_steps: int = 3000
    phase2_steps: int = 9000
    val_every: int = 1000
    lr: float = 1e-6
    clip_min: float = -1.0
    clip_max: float = 1.0
    ema_decay: float = 0.999
    bn_batches: int = 64
    backend: str = "qnnpack"


@dataclass
class DeployConfig:
    input_h: int = 360
    input_w: int = 640
    onnx_output: str = "mobileone_sr_x3.onnx"
    rknn_output: str = "mobileone_sr_x3.rknn"
    calib_dir: str = "data/rknn_calib.txt"
    rknn_python: str = ".venv-rknn/bin/python"
    rknn_quantize: str = "kl_divergence"
    rknn_quantized_method: str = "channel"
    rknn_encrypt: bool = False
    rknn_crypt_level: int = 1


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3_qat: Stage3QatConfig = field(default_factory=Stage3QatConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load YAML config from *path*, or the bundled default when omitted."""
    if path is None:
        config_text = (
            resources.files("rk3588_mobile_sr.config")
            .joinpath("mobileone_sr_x3.yaml")
            .read_text(encoding="utf-8")
        )
        raw = yaml.safe_load(config_text) or {}
    else:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    return AppConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        stage1=_merge_dataclass(Stage1Config, raw.get("stage1")),
        stage2=_merge_dataclass(Stage2Config, raw.get("stage2")),
        stage3_qat=_merge_dataclass(Stage3QatConfig, raw.get("stage3_qat")),
        deploy=_merge_dataclass(DeployConfig, raw.get("deploy")),
    )


def default_config_path() -> Path:
    """Return the path to the bundled default config file."""
    return Path(str(resources.files("rk3588_mobile_sr.config").joinpath("mobileone_sr_x3.yaml")))
