"""YAML configuration loading for PhaseRLFNSR."""

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
    num_blocks: int = 4
    phase_factor: int = 2
    codec_feature_channels: int = 96
    codec_project_channels: int = 16
    codec_upsample_factor: int = 4
    negative_slope: float = 0.1


@dataclass
class DataConfig:
    dataset_description: str = "data/OpenVidHD/openvidhd_60k64_frame_sequences.csv"
    video_root: str = "data/OpenVidHD"
    mlvc_repo: str = "third_party/mlvc"
    mlvc_checkpoint: str = "data/mlvc/mlvc-s-psnr-v1.ckpt"
    mlvc_variant: str = "small"
    sequence_frames: int = 8
    q_indices: tuple[int, ...] = (0, 21, 42, 63)
    val_fraction: float = 0.01
    val_samples: int = 16
    split_seed: int = 42
    lr_size: tuple[int, int] = (360, 640)
    hr_size: tuple[int, int] = (1080, 1920)
    colorspace: str = "yuv"
    augment: bool = True
    num_workers: int = 4
    prefetch_batches: int = 1
    mlvc_amp: bool = True
    codec_context: bool = True
    codec_dropout: float = 0.25


@dataclass
class TrainingConfig:
    batch_size: int = 2
    qat_batch_size: int = 1
    log_every: int = 500
    val_every: int = 1000
    save_every: int = 5000
    float_lr: float = 1e-3
    qat_lr: float = 1e-5
    float_patience: int = 10
    float_min_delta: float = 0.1
    float_min_evaluations: int = 12
    float_safety_max_steps: int = 100_000
    observer_patience: int = 3
    observer_min_delta: float = 0.02
    observer_min_evaluations: int = 5
    observer_safety_max_steps: int = 15_000
    qat_patience: int = 5
    qat_min_delta: float = 0.02
    qat_min_evaluations: int = 7
    qat_safety_max_steps: int = 30_000
    clip_min: float = -1.0
    clip_max: float = 1.0
    ema_decay: float = 0.999
    backend: str = "qnnpack"
    val_metric: str = "vmaf"


@dataclass
class DeployConfig:
    target: str = "rk3576"
    input_h: int = 360
    input_w: int = 640
    onnx_output: str = "phase_rlfn_sr_x3.onnx"
    rknn_output: str = "phase_rlfn_sr_x3.rknn"
    calib_dir: str = "data/rknn_calib.txt"
    rknn_python: str = ".venv-rknn/bin/python"
    rknn_quantize: str = "kl_divergence"
    rknn_quantized_method: str = "channel"
    rknn_encrypt: bool = False
    rknn_crypt_level: int = 1
    codec_context: bool = True

    def __post_init__(self) -> None:
        # Import lazily so the general configuration module does not otherwise
        # depend on deployment implementation details.
        from rknn_super_resolution.deploy.targets import normalize_rknn_target

        self.target = normalize_rknn_target(self.target)


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - valid
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {cls.__name__} field(s): {names}")
    kwargs = dict(data)
    if cls is DataConfig and "lr_size" in kwargs:
        kwargs["lr_size"] = tuple(kwargs["lr_size"])
    if cls is DataConfig and "hr_size" in kwargs:
        kwargs["hr_size"] = tuple(kwargs["hr_size"])
    if cls is DataConfig and "q_indices" in kwargs:
        kwargs["q_indices"] = tuple(kwargs["q_indices"])
    return cls(**kwargs)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load YAML config from *path*, or the bundled default when omitted."""
    if path is None:
        config_text = (
            resources.files("rknn_super_resolution.config")
            .joinpath("phase_rlfn_sr_x3.yaml")
            .read_text(encoding="utf-8")
        )
        raw = yaml.safe_load(config_text) or {}
    else:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    sections = {"model", "data", "training", "deploy"}
    unknown_sections = set(raw) - sections
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unknown config section(s): {names}")

    return AppConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        deploy=_merge_dataclass(DeployConfig, raw.get("deploy")),
    )


def default_config_path() -> Path:
    """Return the path to the bundled default config file."""
    return Path(
        str(resources.files("rknn_super_resolution.config").joinpath("phase_rlfn_sr_x3.yaml"))
    )
