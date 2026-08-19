"""Canvas codec training loader (offline LR/HR .npy, pure CPU mmap)."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F

from rk3588_mobile_sr.data.augment import (
    augment_config_for_canvas,
    batch_augment_pair,
    batch_random_crop_pair,
    sample_lr_crop_xy,
)
from rk3588_mobile_sr.data.codec_index import CodecFrameEntry, build_codec_frame_index
from rk3588_mobile_sr.data.yuv_utils import rgb_to_model_colorspace


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
    augment_lr_decode_noise: bool = False,
    augment_lr_jpeg: bool = False,
    pre_cropped: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize, optionally crop on CPU, then augment/colorspace on ``device``.

    When ``pre_cropped=True``, ``lr``/``hr`` are already patch-sized and the
    random-crop step is skipped.
    """
    lr = uint8_bchw_to_float(lr)
    hr = uint8_bchw_to_float(hr)

    lr_h, lr_w = lr_size
    hr_h, hr_w = hr_size
    if not pre_cropped:
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
    elif patch_size is not None:
        lr_h = lr_w = patch_size
        hr_h = hr_w = patch_size * scale

    lr = lr.to(device, non_blocking=True)
    hr = hr.to(device, non_blocking=True)

    if augment:
        lr, hr = batch_augment_pair(
            lr,
            hr,
            config=augment_config_for_canvas(
                augment_rot90=augment_rot90,
                patch_size=patch_size,
                augment_lr_decode_noise=augment_lr_decode_noise,
                augment_lr_jpeg=augment_lr_jpeg,
            ),
            lr_canvas=(lr_h, lr_w),
            hr_canvas=(hr_h, hr_w),
        )

    lr = rgb_to_model_colorspace(lr, colorspace=colorspace, nv12_simulate=nv12_simulate)
    hr = rgb_to_model_colorspace(hr, colorspace=colorspace, nv12_simulate=nv12_simulate)

    return lr, hr


def resolve_decode_backend(requested: str) -> str:
    """Resolve the decode backend. Runtime reads offline LR/HR .npy only."""
    if requested in ("raw", "auto"):
        return "raw"
    raise ValueError(f"Unsupported decode {requested!r}; use auto or raw")


@dataclass(frozen=True)
class TrainDataSettings:
    codec_manifest: str
    lr_size: tuple[int, int]
    hr_size: tuple[int, int]
    colorspace: str = "yuv"
    nv12_simulate: bool = True
    augment: bool = True
    decode: str = "auto"
    decode_num_workers: int = 4
    project_root: str | None = None
    patch_size: int | None = None
    scale: int = 3
    augment_rot90: bool = False
    augment_lr_decode_noise: bool = False
    augment_lr_jpeg: bool = False


@dataclass
class CodecTrainLoader:
    """Training loader bundle consumed by StepTrainer."""

    dataloader: Iterator[tuple[torch.Tensor, torch.Tensor]]
    _close: object | None = None

    def close(self) -> None:
        if self._close is not None and hasattr(self._close, "close"):
            self._close.close()  # type: ignore[union-attr]


class RawFrameTrainIterator(Iterator[tuple[torch.Tensor, torch.Tensor]]):
    """Infinite iterator: mmap LR/HR .npy, optional windowed crop, GPU augment."""

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
        self._settings = settings
        self._batch_size = batch_size
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{device_id}")
        else:
            self._device = torch.device("cpu")

        index = build_codec_frame_index(
            manifest,
            project_root=project_root,
            seed=seed,
        )
        shard = max(world_size, 1)
        self._entries = index.entries[rank::shard]
        if not self._entries:
            raise ValueError(
                f"rank {rank}/{shard} received zero codec frames after sharding"
            )
        self._pos = 0
        self._npy_cache: dict[Path, np.ndarray] = {}
        self._cache_lock = Lock()
        workers = max(0, int(settings.decode_num_workers))
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        )

    def __iter__(self) -> RawFrameTrainIterator:
        return self

    def _mmap_npy(self, path: Path) -> np.ndarray:
        with self._cache_lock:
            cached = self._npy_cache.get(path)
            if cached is None:
                cached = np.load(path, mmap_mode="r")
                self._npy_cache[path] = cached
            return cached

    def _read_pair(self, entry: CodecFrameEntry) -> tuple[torch.Tensor, torch.Tensor]:
        lr_clip = self._mmap_npy(entry.lr_path)
        hr_clip = self._mmap_npy(entry.hr_path)
        lr_frame = lr_clip[entry.lr_frame]
        hr_frame = hr_clip[entry.hr_frame]
        patch = self._settings.patch_size
        scale = self._settings.scale

        if patch is not None:
            lr_h, lr_w = int(lr_frame.shape[0]), int(lr_frame.shape[1])
            hr_h, hr_w = int(hr_frame.shape[0]), int(hr_frame.shape[1])
            if hr_h != lr_h * scale or hr_w != lr_w * scale:
                raise ValueError(
                    f"windowed read requires HR==LR*scale, got LR {lr_h}x{lr_w}, "
                    f"HR {hr_h}x{hr_w}, scale={scale} for {entry.record_id!r}"
                )
            top, left = sample_lr_crop_xy(
                lr_h,
                lr_w,
                lr_crop_h=patch,
                lr_crop_w=patch,
                scale=scale,
            )
            lr_np = np.array(lr_frame[top : top + patch, left : left + patch], copy=True)
            hr_np = np.array(
                hr_frame[
                    top * scale : (top + patch) * scale,
                    left * scale : (left + patch) * scale,
                ],
                copy=True,
            )
            lr = torch.from_numpy(lr_np).permute(2, 0, 1).contiguous().float()
            hr = torch.from_numpy(hr_np).permute(2, 0, 1).contiguous().float()
            return lr, hr

        lr = torch.from_numpy(np.array(lr_frame, copy=True)).permute(2, 0, 1).contiguous().float()
        hr = torch.from_numpy(np.array(hr_frame, copy=True)).permute(2, 0, 1).contiguous().float()
        return lr, hr

    def _next_entries(self) -> list[CodecFrameEntry]:
        n = len(self._entries)
        out: list[CodecFrameEntry] = []
        for _ in range(self._batch_size):
            out.append(self._entries[self._pos % n])
            self._pos += 1
        return out

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        entries = self._next_entries()
        if self._pool is not None:
            pairs = list(self._pool.map(self._read_pair, entries))
        else:
            pairs = [self._read_pair(e) for e in entries]
        lr = torch.stack([p[0] for p in pairs], dim=0)
        hr = torch.stack([p[1] for p in pairs], dim=0)
        pre_cropped = self._settings.patch_size is not None
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
            augment_lr_decode_noise=self._settings.augment_lr_decode_noise,
            augment_lr_jpeg=self._settings.augment_lr_jpeg,
            pre_cropped=pre_cropped,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._npy_cache.clear()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_codec_train_loader(
    settings: TrainDataSettings,
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    device_id: int | None = None,
) -> CodecTrainLoader:
    """Build the canvas codec training iterator (LR/HR .npy)."""
    root = Path(settings.project_root or project_root())
    dev_id = rank if device_id is None else device_id
    backend = resolve_decode_backend(settings.decode)
    if backend != "raw":
        raise RuntimeError(f"only raw backend supported, got {backend!r}")
    iterator = RawFrameTrainIterator(
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
        decode_num_workers=getattr(data, "decode_num_workers", 4),
        project_root=str(project_root()),
        patch_size=getattr(args, "patch_size", None),
        scale=getattr(args, "scale", 3),
        augment_rot90=getattr(data, "augment_rot90", False),
        augment_lr_decode_noise=getattr(data, "augment_lr_decode_noise", False),
        augment_lr_jpeg=getattr(data, "augment_lr_jpeg", False),
        nv12_simulate=getattr(data, "nv12_simulate", True)
        if not getattr(args, "no_nv12_simulate", False)
        else False,
    )
