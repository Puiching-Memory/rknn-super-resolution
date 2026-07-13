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
    negative_slope: float = 0.1


@dataclass
class DataConfig:
    codec_manifest: str = "data/codec_cache/manifest.jsonl"
    val_manifest: str = "data/sources/manifests/val_fixed.jsonl"
    lr_size: tuple[int, int] = (360, 640)
    hr_size: tuple[int, int] = (1080, 1920)
    colorspace: str = "yuv"
    nv12_simulate: bool = True
    augment: bool = True
    decode: str = "auto"
    dali_num_threads: int = 4
    dali_initial_fill: int = 32
    prefetch_batches: int = 4
    augment_rot90: bool = False
    augment_lr_decode_noise: bool = False
    augment_lr_jpeg: bool = False


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
class Stage2QatConfig:
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
    stage1_weight: str = "checkpoints/stage1/best.pth"


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
    stage2_qat: Stage2QatConfig = field(default_factory=Stage2QatConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in data.items() if k in valid}
    if cls is DataConfig and "lr_size" in kwargs:
        kwargs["lr_size"] = tuple(kwargs["lr_size"])
    if cls is DataConfig and "hr_size" in kwargs:
        kwargs["hr_size"] = tuple(kwargs["hr_size"])
    return cls(**kwargs)


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
        stage2_qat=_merge_dataclass(Stage2QatConfig, raw.get("stage2_qat")),
        deploy=_merge_dataclass(DeployConfig, raw.get("deploy")),
    )


def default_config_path() -> Path:
    """Return the path to the bundled default config file."""
    return Path(str(resources.files("rk3588_mobile_sr.config").joinpath("mobileone_sr_x3.yaml")))
