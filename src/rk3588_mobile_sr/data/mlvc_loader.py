"""OpenVidHD loaders with frozen MLVC-S reconstruction on GPU."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from rk3588_mobile_sr.config import DataConfig
from rk3588_mobile_sr.data.decode import FrameDecoder, TorchCodecFrameDecoder
from rk3588_mobile_sr.data.mlvc_runtime import FrozenMLVCRuntime
from rk3588_mobile_sr.data.openvid import (
    OpenVidSequenceDataset,
    collate_openvid_batch,
    load_openvid_description,
    split_sequence_indices,
)


class MLVCRuntime(Protocol):
    def reconstruct(self, sequence: torch.Tensor, q_index: torch.Tensor) -> torch.Tensor: ...


def rgb_to_mlvc_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to the BT.709 full-range representation used by MLVC."""
    r, g, b = rgb.chunk(3, dim=-3)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = 0.5 * (b - y) / (1.0 - 0.0722) + 0.5
    cr = 0.5 * (r - y) / (1.0 - 0.2126) + 0.5
    return torch.cat((y, cb, cr), dim=-3).clamp_(0.0, 1.0)


def mlvc_ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert MLVC BT.709 full-range YCbCr [0,1] to RGB [0,1]."""
    y, cb, cr = ycbcr.chunk(3, dim=-3)
    r = y + (2.0 - 2.0 * 0.2126) * (cr - 0.5)
    b = y + (2.0 - 2.0 * 0.0722) * (cb - 0.5)
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return torch.cat((r, g, b), dim=-3).clamp_(0.0, 1.0)


class MLVCBatchProcessor:
    def __init__(
        self,
        runtime: MLVCRuntime,
        *,
        decoder: FrameDecoder,
        device: torch.device,
        q_indices: tuple[int, ...],
        colorspace: str,
        patch_size: int | None,
        scale: int,
    ) -> None:
        if not q_indices or any(q < 0 or q >= 64 for q in q_indices):
            raise ValueError("q_indices must contain values in [0, 63]")
        if colorspace not in ("rgb", "yuv"):
            raise ValueError("colorspace must be 'rgb' or 'yuv'")
        self.runtime = runtime
        self.decoder = decoder
        self.device = device
        self.q_indices = q_indices
        self.colorspace = colorspace
        self.patch_size = patch_size
        self.scale = scale

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_u8, hr_u8 = self.decoder.decode_batch(batch)
        sequence = sequence_u8.to(self.device, non_blocking=True).float().div_(255.0)
        hr_rgb = hr_u8.to(self.device, non_blocking=True).float().div_(255.0)
        b, t, c, h, w = sequence.shape
        sequence_yuv = rgb_to_mlvc_ycbcr(sequence.reshape(b * t, c, h, w)).reshape(
            b, t, c, h, w
        )

        if training:
            choices = torch.tensor(self.q_indices, device=self.device)
            offsets = torch.randint(len(self.q_indices), (b,), device=self.device)
            q_index = choices[offsets]
        else:
            q_index = batch["q_index"].to(self.device, non_blocking=True)

        lr_yuv = self.runtime.reconstruct(sequence_yuv, q_index)
        hr_yuv = rgb_to_mlvc_ycbcr(hr_rgb)
        if self.colorspace == "rgb":
            lr = mlvc_ycbcr_to_rgb(lr_yuv)
            hr = hr_rgb
        else:
            lr = lr_yuv
            hr = hr_yuv
        lr = lr.mul(255.0)
        hr = hr.mul(255.0)

        if self.patch_size is not None:
            lr, hr = self._random_crop(lr, hr)
        return lr.contiguous(), hr.contiguous()

    def _random_crop(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patch_size
        assert patch is not None
        if hr.shape[-2:] != (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale):
            raise ValueError("MLVC LR and OpenVidHD HR canvases are not scale-aligned")
        if patch > lr.shape[-2] or patch > lr.shape[-1]:
            raise ValueError(f"patch_size {patch} exceeds LR canvas {tuple(lr.shape[-2:])}")
        lr_out: list[torch.Tensor] = []
        hr_out: list[torch.Tensor] = []
        for index in range(lr.shape[0]):
            top = int(torch.randint(lr.shape[-2] - patch + 1, (), device=lr.device))
            left = int(torch.randint(lr.shape[-1] - patch + 1, (), device=lr.device))
            lr_out.append(lr[index, :, top : top + patch, left : left + patch])
            hr_top, hr_left = top * self.scale, left * self.scale
            hr_patch = patch * self.scale
            hr_out.append(
                hr[index, :, hr_top : hr_top + hr_patch, hr_left : hr_left + hr_patch]
            )
        return torch.stack(lr_out), torch.stack(hr_out)


@dataclass
class MLVCDeviceBatch:
    lr: torch.Tensor
    hr: torch.Tensor
    ready_event: torch.cuda.Event | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.lr
        yield self.hr

    def wait_ready(self, stream: torch.cuda.Stream | None = None) -> None:
        if self.ready_event is None:
            return
        consumer = stream or torch.cuda.current_stream(self.lr.device)
        consumer.wait_event(self.ready_event)


class _InfiniteMLVCIterator(Iterator[MLVCDeviceBatch]):
    def __init__(
        self,
        loader: DataLoader,
        processor: MLVCBatchProcessor,
        sampler: DistributedSampler | None,
    ) -> None:
        self.loader = loader
        self.processor = processor
        self.sampler = sampler
        self.epoch = 0
        self.iterator = iter(loader)

    def __iter__(self) -> _InfiniteMLVCIterator:
        return self

    def __next__(self) -> MLVCDeviceBatch:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        lr, hr = self.processor(batch, training=True)
        ready_event = None
        if lr.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(lr.device))
        return MLVCDeviceBatch(lr, hr, ready_event)

    def close(self) -> None:
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


@dataclass
class MLVCTrainLoader:
    dataloader: _InfiniteMLVCIterator
    decoder: FrameDecoder | None = None

    def __iter__(self) -> Iterator[MLVCDeviceBatch]:
        return iter(self.dataloader)

    def close(self) -> None:
        self.dataloader.close()
        if self.decoder is not None:
            self.decoder.close()


class MLVCValidationLoader:
    def __init__(
        self,
        loader: DataLoader,
        processor: MLVCBatchProcessor,
    ) -> None:
        self.loader = loader
        self.processor = processor
        self.dataset = loader.dataset
        self.sampler = loader.sampler

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.loader:
            yield self.processor(batch, training=False)

    def __len__(self) -> int:
        return len(self.loader)


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def build_mlvc_loaders(
    data: DataConfig,
    *,
    device: torch.device,
    batch_size: int,
    patch_size: int | None,
    scale: int,
    colorspace: str,
    train_aug: bool,
    val_batch_size: int,
    rank: int,
    world_size: int,
    project_root: Path,
) -> tuple[MLVCTrainLoader, MLVCValidationLoader]:
    description = _resolve(data.dataset_description, project_root)
    sequences = load_openvid_description(description)
    train_indices, val_indices = split_sequence_indices(
        len(sequences), val_fraction=data.val_fraction, seed=data.split_seed
    )
    train_dataset = OpenVidSequenceDataset(
        description,
        indices=train_indices,
        sequence_frames=data.sequence_frames,
        lr_size=data.lr_size,
        hr_size=data.hr_size,
        training_split=True,
        augment=train_aug,
    )
    val_base_count = max(1, data.val_samples // len(data.q_indices))
    val_dataset = OpenVidSequenceDataset(
        description,
        indices=val_indices,
        sequence_frames=data.sequence_frames,
        lr_size=data.lr_size,
        hr_size=data.hr_size,
        training_split=False,
        augment=False,
        q_indices=data.q_indices,
        max_samples=val_base_count,
    )

    distributed = world_size > 1 and dist.is_initialized()
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    worker_args = {
        "num_workers": data.num_workers,
        "pin_memory": True,
        "persistent_workers": data.num_workers > 0,
        "collate_fn": collate_openvid_batch,
    }
    train_cpu_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **worker_args,
    )
    if len(train_cpu_loader) == 0:
        raise ValueError(
            f"OpenVidHD train split is too small for batch_size={batch_size} on rank {rank}"
        )
    val_cpu_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **worker_args,
    )

    runtime = FrozenMLVCRuntime(
        repo=_resolve(data.mlvc_repo, project_root),
        checkpoint=_resolve(data.mlvc_checkpoint, project_root),
        variant=data.mlvc_variant,
        device=device,
        amp=data.mlvc_amp,
    )
    decoder = TorchCodecFrameDecoder(
        device,
        lr_size=data.lr_size,
        hr_size=data.hr_size,
    )
    train_processor = MLVCBatchProcessor(
        runtime,
        decoder=decoder,
        device=device,
        q_indices=data.q_indices,
        colorspace=colorspace,
        patch_size=patch_size,
        scale=scale,
    )
    val_processor = MLVCBatchProcessor(
        runtime,
        decoder=decoder,
        device=device,
        q_indices=data.q_indices,
        colorspace=colorspace,
        patch_size=None,
        scale=scale,
    )
    train_iterator = _InfiniteMLVCIterator(train_cpu_loader, train_processor, train_sampler)
    return (
        MLVCTrainLoader(train_iterator, decoder),
        MLVCValidationLoader(val_cpu_loader, val_processor),
    )
