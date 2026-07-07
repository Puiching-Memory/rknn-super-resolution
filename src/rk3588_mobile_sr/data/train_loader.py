"""Canvas codec training loader (DALI NVDEC + TorchCodec fallback)."""

from __future__ import annotations

import ctypes
import ctypes.util
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from rk3588_mobile_sr.data.augment import (
    augment_config_for_canvas,
    batch_augment_pair,
    batch_random_crop_pair,
)
from rk3588_mobile_sr.data.codec_index import CodecFrameEntry, build_codec_frame_index
from rk3588_mobile_sr.data.yuv_utils import rgb_to_model_colorspace

_TORCHCODEC_AVAILABLE = False
try:
    from torchcodec.decoders import VideoDecoder

    _TORCHCODEC_AVAILABLE = True
except ImportError:
    VideoDecoder = None  # type: ignore[misc, assignment]


def uint8_bchw_to_float(batch: torch.Tensor) -> torch.Tensor:
    """Convert BCHW or NFHWC uint8/float tensors to BCHW float32 in [0, 255]."""
    if batch.ndim == 5:
        batch = batch[:, 0]
    if batch.ndim == 4 and batch.shape[-1] == 3:
        batch = batch.permute(0, 3, 1, 2).contiguous()
    if batch.dtype != torch.uint8:
        return batch.float()
    return batch.to(dtype=torch.float32)


def apply_canvas_batch_transform(
    lr: torch.Tensor,
    hr: torch.Tensor,
    *,
    lr_size: tuple[int, int],
    hr_size: tuple[int, int],
    colorspace: str,
    nv12_simulate: bool = False,
    augment: bool,
    device: torch.device,
    patch_size: int | None = None,
    scale: int = 3,
    augment_rot90: bool = False,
    augment_lr_blur: bool = False,
    augment_lr_motion_blur: bool = False,
    augment_lr_noise: bool = False,
    augment_lr_jpeg: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize, optionally crop, augment, and convert decoded RGB batches."""
    lr = uint8_bchw_to_float(lr).to(device, non_blocking=True)
    hr = uint8_bchw_to_float(hr).to(device, non_blocking=True)

    lr_h, lr_w = lr_size
    hr_h, hr_w = hr_size
    if hr.shape[-2] != hr_h or hr.shape[-1] != hr_w:
        hr = F.interpolate(hr, size=(hr_h, hr_w), mode="area")
    if lr.shape[-2] != lr_h or lr.shape[-1] != lr_w:
        lr = F.interpolate(lr, size=(lr_h, lr_w), mode="area")

    if patch_size is not None:
        lr, hr = batch_random_crop_pair(
            lr,
            hr,
            lr_crop_h=patch_size,
            lr_crop_w=patch_size,
            scale=scale,
        )
        lr_h = lr_w = patch_size
        hr_h = hr_w = patch_size * scale

    if augment:
        lr, hr = batch_augment_pair(
            lr,
            hr,
            config=augment_config_for_canvas(
                augment_rot90=augment_rot90,
                patch_size=patch_size,
                augment_lr_blur=augment_lr_blur,
                augment_lr_motion_blur=augment_lr_motion_blur,
                augment_lr_noise=augment_lr_noise,
                augment_lr_jpeg=augment_lr_jpeg,
            ),
            lr_canvas=(lr_h, lr_w),
            hr_canvas=(hr_h, hr_w),
        )

    lr = rgb_to_model_colorspace(lr, colorspace=colorspace, nv12_simulate=nv12_simulate)
    hr = rgb_to_model_colorspace(hr, colorspace=colorspace, nv12_simulate=nv12_simulate)

    return lr, hr


def nvidia_cuvid_available() -> bool:
    """Return True when libnvcuvid is loadable (required by DALI VideoReader)."""
    path = ctypes.util.find_library("nvcuvid")
    if path:
        try:
            ctypes.CDLL(path)
            return True
        except OSError:
            pass
    for name in ("libnvcuvid.so.1", "libnvcuvid.so"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


@pipeline_def
def _codec_pair_pipeline(
    lr_file_list: str,
    hr_file_list: str,
    *,
    shard_id: int,
    num_shards: int,
    initial_fill: int,
    reader_seed: int,
):
    lr, _lr_lbl = fn.readers.video(
        device="gpu",
        name="lr_reader",
        file_list=lr_file_list,
        sequence_length=1,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        initial_fill=initial_fill,
        file_list_frame_num=True,
        enable_frame_num="none",
        enable_timestamps=False,
        dtype=types.UINT8,
        seed=reader_seed,
    )
    hr, _hr_lbl = fn.readers.video(
        device="gpu",
        name="hr_reader",
        file_list=hr_file_list,
        sequence_length=1,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        initial_fill=initial_fill,
        file_list_frame_num=True,
        enable_frame_num="none",
        enable_timestamps=False,
        dtype=types.UINT8,
        seed=reader_seed,
    )
    return lr, hr


class DaliCodecTrainIterator(Iterator[tuple[torch.Tensor, torch.Tensor]]):
    """Infinite iterator over DALI-decoded LR/HR canvas batches."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        project_root: Path,
        settings: TrainDataSettings,
        batch_size: int,
        device_id: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if not nvidia_cuvid_available():
            raise RuntimeError(
                "DALI video decode requires libnvcuvid.so (NVDEC). "
                "Set NVIDIA_DRIVER_CAPABILITIES=compute,utility,video or use decode=torchcodec."
            )

        self._settings = settings
        self._device = torch.device(f"cuda:{device_id}")
        self._index = build_codec_frame_index(
            manifest,
            project_root=project_root,
            seed=seed + rank,
            for_dali=True,
        )
        assert self._index.lr_list is not None and self._index.hr_list is not None
        self._pipe = _codec_pair_pipeline(
            batch_size=batch_size,
            num_threads=settings.dali_num_threads,
            device_id=device_id,
            lr_file_list=str(self._index.lr_list),
            hr_file_list=str(self._index.hr_list),
            shard_id=rank,
            num_shards=max(world_size, 1),
            initial_fill=settings.dali_initial_fill,
            reader_seed=seed + rank,
            seed=seed + rank,
        )
        self._pipe.build()
        self._iterator = DALIGenericIterator(
            [self._pipe],
            output_map=["lr", "hr"],
            reader_name="lr_reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def __iter__(self) -> DaliCodecTrainIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            batch = next(self._iterator)
            lr = batch[0]["lr"]
            hr = batch[0]["hr"]
            if lr.shape[0] == 0:
                continue
            return apply_canvas_batch_transform(
                lr,
                hr,
                lr_size=self._settings.lr_size,
                hr_size=self._settings.hr_size,
                colorspace=self._settings.colorspace,
                nv12_simulate=self._settings.nv12_simulate,
                augment=self._settings.augment,
                device=self._device,
                patch_size=self._settings.patch_size,
                scale=self._settings.scale,
                augment_rot90=self._settings.augment_rot90,
                augment_lr_blur=self._settings.augment_lr_blur,
                augment_lr_motion_blur=self._settings.augment_lr_motion_blur,
                augment_lr_noise=self._settings.augment_lr_noise,
                augment_lr_jpeg=self._settings.augment_lr_jpeg,
            )

    def close(self) -> None:
        if self._index.temp_dir is not None:
            self._index.temp_dir.cleanup()


def _frame_to_float_rgb(frame: torch.Tensor) -> torch.Tensor:
    out = frame.float() if frame.dtype != torch.float32 else frame
    if out.numel() > 0 and out.max() <= 1.0:
        out = out * 255.0
    return out


class TorchCodecTrainIterator(Iterator[tuple[torch.Tensor, torch.Tensor]]):
    """Infinite iterator decoding codec frames on CPU, post-processing on GPU."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        project_root: Path,
        settings: TrainDataSettings,
        batch_size: int,
        device_id: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if VideoDecoder is None:
            raise RuntimeError("torchcodec is required for CPU decode fallback")

        self._settings = settings
        self._batch_size = batch_size
        self._device = torch.device(f"cuda:{device_id}")
        self._world_size = world_size
        index = build_codec_frame_index(
            manifest,
            project_root=project_root,
            seed=seed + rank,
            for_dali=False,
        )
        self._entries = index.entries
        self._pos = rank % max(len(self._entries), 1)
        self._lr_decoders: dict[Path, object] = {}
        self._hr_decoders: dict[Path, object] = {}

    def _decoder(self, path: Path, cache: dict[Path, object]) -> VideoDecoder:
        if path not in cache:
            cache[path] = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=0)
        return cache[path]  # type: ignore[return-value]

    def _next_batch_entries(self) -> list[CodecFrameEntry]:
        n = len(self._entries)
        if n == 0:
            raise RuntimeError("codec manifest produced zero training frames")
        batch: list[CodecFrameEntry] = []
        for _ in range(self._batch_size):
            batch.append(self._entries[self._pos % n])
            self._pos += self._world_size
        return batch

    def __iter__(self) -> TorchCodecTrainIterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        entries = self._next_batch_entries()
        by_lr: dict[Path, list[tuple[int, int]]] = defaultdict(list)
        by_hr: dict[Path, list[tuple[int, int]]] = defaultdict(list)
        for out_idx, entry in enumerate(entries):
            by_lr[entry.lr_path].append((entry.lr_frame, out_idx))
            by_hr[entry.hr_path].append((entry.hr_frame, out_idx))

        lr_frames = [torch.empty(0)] * len(entries)
        hr_frames = [torch.empty(0)] * len(entries)

        for path, items in by_lr.items():
            dec = self._decoder(path, self._lr_decoders)
            indices = [i for i, _ in items]
            batch = dec.get_frames_at(indices)
            for (frame_idx, out_idx), tensor in zip(items, batch.data, strict=True):
                del frame_idx
                lr_frames[out_idx] = _frame_to_float_rgb(tensor)

        for path, items in by_hr.items():
            dec = self._decoder(path, self._hr_decoders)
            indices = [i for i, _ in items]
            batch = dec.get_frames_at(indices)
            for (frame_idx, out_idx), tensor in zip(items, batch.data, strict=True):
                del frame_idx
                hr_frames[out_idx] = _frame_to_float_rgb(tensor)

        lr = torch.stack(lr_frames, dim=0)
        hr = torch.stack(hr_frames, dim=0)
        lr_u8 = lr.clamp(0, 255).to(torch.uint8)
        hr_u8 = hr.clamp(0, 255).to(torch.uint8)
        return apply_canvas_batch_transform(
            lr_u8,
            hr_u8,
            lr_size=self._settings.lr_size,
            hr_size=self._settings.hr_size,
            colorspace=self._settings.colorspace,
            nv12_simulate=self._settings.nv12_simulate,
            augment=self._settings.augment,
            device=self._device,
            patch_size=self._settings.patch_size,
            scale=self._settings.scale,
            augment_rot90=self._settings.augment_rot90,
            augment_lr_blur=self._settings.augment_lr_blur,
            augment_lr_motion_blur=self._settings.augment_lr_motion_blur,
            augment_lr_noise=self._settings.augment_lr_noise,
            augment_lr_jpeg=self._settings.augment_lr_jpeg,
        )

    def close(self) -> None:
        self._lr_decoders.clear()
        self._hr_decoders.clear()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_decode_backend(requested: str) -> str:
    """Pick DALI or TorchCodec from config."""
    if requested == "dali":
        if not nvidia_cuvid_available():
            raise RuntimeError("decode=dali requires libnvcuvid.so (NVDEC)")
        if not torch.cuda.is_available():
            raise RuntimeError("decode=dali requires CUDA")
        return "dali"
    if requested == "torchcodec":
        return "torchcodec"
    if requested == "auto":
        if torch.cuda.is_available() and nvidia_cuvid_available():
            return "dali"
        return "torchcodec"
    raise ValueError(f"Unsupported decode {requested!r}; use auto, dali, or torchcodec")


@dataclass(frozen=True)
class TrainDataSettings:
    codec_manifest: str
    lr_size: tuple[int, int]
    hr_size: tuple[int, int]
    colorspace: str = "yuv"
    nv12_simulate: bool = True
    augment: bool = True
    decode: str = "auto"
    dali_num_threads: int = 4
    dali_initial_fill: int = 32
    project_root: str | None = None
    patch_size: int | None = None
    scale: int = 3
    augment_rot90: bool = False
    augment_lr_blur: bool = False
    augment_lr_motion_blur: bool = False
    augment_lr_noise: bool = False
    augment_lr_jpeg: bool = False


@dataclass
class CodecTrainLoader:
    """Training loader bundle consumed by StepTrainer."""

    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]]
    _close: object | None = None

    def close(self) -> None:
        if self._close is not None and hasattr(self._close, "close"):
            self._close.close()  # type: ignore[union-attr]


def build_codec_train_loader(
    settings: TrainDataSettings,
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    device_id: int | None = None,
) -> CodecTrainLoader:
    """Build the canvas codec training iterator."""
    root = Path(settings.project_root or project_root())
    dev_id = rank if device_id is None else device_id
    backend = resolve_decode_backend(settings.decode)

    if backend == "dali":
        iterator = DaliCodecTrainIterator(
            settings.codec_manifest,
            project_root=root,
            settings=settings,
            batch_size=batch_size,
            device_id=dev_id,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
    else:
        if settings.decode == "auto" and not nvidia_cuvid_available():
            warnings.warn(
                "NVDEC unavailable; falling back to TorchCodec CPU decode.",
                stacklevel=2,
            )
        iterator = TorchCodecTrainIterator(
            settings.codec_manifest,
            project_root=root,
            settings=settings,
            batch_size=batch_size,
            device_id=dev_id,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )

    return CodecTrainLoader(dataloader=iterator, _close=iterator)


def data_settings_from_args(args) -> TrainDataSettings:
    """Build TrainDataSettings from CLI args and YAML."""
    from rk3588_mobile_sr.config import load_config

    app_cfg = load_config(getattr(args, "config", None))
    data = app_cfg.data

    manifest = getattr(args, "codec_manifest", None) or data.codec_manifest
    if not manifest:
        raise ValueError("--codec_manifest or data.codec_manifest is required")

    decode = getattr(args, "decode", None) or data.decode
    augment = getattr(args, "train_aug", True)

    return TrainDataSettings(
        codec_manifest=manifest,
        lr_size=tuple(data.lr_size),
        hr_size=tuple(data.hr_size),
        colorspace=getattr(args, "colorspace", None) or data.colorspace,
        augment=augment,
        decode=decode,
        dali_num_threads=data.dali_num_threads,
        dali_initial_fill=data.dali_initial_fill,
        project_root=str(project_root()),
        patch_size=getattr(args, "patch_size", None),
        scale=getattr(args, "scale", 3),
        augment_rot90=getattr(data, "augment_rot90", False),
        augment_lr_blur=data.augment_lr_blur,
        augment_lr_motion_blur=data.augment_lr_motion_blur,
        augment_lr_noise=data.augment_lr_noise,
        augment_lr_jpeg=data.augment_lr_jpeg,
        nv12_simulate=getattr(data, "nv12_simulate", True)
        if not getattr(args, "no_nv12_simulate", False)
        else False,
    )
