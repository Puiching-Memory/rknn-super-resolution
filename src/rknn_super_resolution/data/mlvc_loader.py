"""OpenVidHD loaders with frozen MLVC-S reconstruction on GPU."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from rknn_super_resolution.config import DataConfig
from rknn_super_resolution.data.decode import FrameDecoder, TorchCodecFrameDecoder
from rknn_super_resolution.data.mlvc_runtime import FrozenMLVCRuntime, MLVCReconstruction
from rknn_super_resolution.data.openvid import (
    OpenVidSequenceDataset,
    collate_openvid_batch,
    load_openvid_index,
    split_sequence_indices,
)
from rknn_super_resolution.models import SRInput


class MLVCRuntime(Protocol):
    def reconstruct(self, sequence: torch.Tensor, q_index: torch.Tensor) -> MLVCReconstruction: ...


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


def _map_time(frames: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply an NCHW color transform across a ``B x T x C x H x W`` clip."""
    batch, time, channels, height, width = frames.shape
    mapped = fn(frames.reshape(batch * time, channels, height, width))
    return mapped.reshape(batch, time, *mapped.shape[1:])


class MLVCBatchProcessor:
    def __init__(
        self,
        runtime: MLVCRuntime,
        *,
        decoder: FrameDecoder,
        device: torch.device,
        q_indices: tuple[int, ...],
        colorspace: str,
        scale: int,
        codec_context: bool = False,
        codec_dropout: float = 0.25,
    ) -> None:
        if not q_indices or any(q < 0 or q >= 64 for q in q_indices):
            raise ValueError("q_indices must contain values in [0, 63]")
        if colorspace not in ("rgb", "yuv"):
            raise ValueError("colorspace must be 'rgb' or 'yuv'")
        if scale < 1:
            raise ValueError("scale must be positive")
        if not 0.0 <= codec_dropout <= 1.0:
            raise ValueError("codec_dropout must be in [0, 1]")
        self.runtime = runtime
        self.decoder = decoder
        self.device = device
        self.q_indices = q_indices
        self.colorspace = colorspace
        self.scale = scale
        self.codec_context = codec_context
        self.codec_dropout = codec_dropout

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        training: bool,
    ) -> tuple[SRInput, torch.Tensor]:
        sequence_u8, hr_u8 = self.decoder.decode_batch(batch)
        if sequence_u8.ndim != 5 or hr_u8.ndim != 5:
            raise ValueError("decoded clips must be BxTx3xHxW")
        sequence = sequence_u8.to(self.device, non_blocking=True).float().div_(255.0)
        hr_rgb = hr_u8.to(self.device, non_blocking=True).float().div_(255.0)
        b, t, c, height, width = sequence.shape
        if hr_rgb.shape[:3] != (b, t, c):
            raise ValueError("LR and HR clips must share batch, time and channel layout")
        if hr_rgb.shape[-2:] != (height * self.scale, width * self.scale):
            raise ValueError("MLVC LR and OpenVidHD HR canvases are not scale-aligned")

        sequence_yuv = _map_time(sequence, rgb_to_mlvc_ycbcr)
        if training:
            choices = torch.tensor(self.q_indices, device=self.device)
            offsets = torch.randint(len(self.q_indices), (b,), device=self.device)
            q_index = choices[offsets]
        else:
            q_index = batch["q_index"].to(self.device, non_blocking=True)

        reconstruction = self.runtime.reconstruct(sequence_yuv, q_index)
        lr_yuv = reconstruction.frames
        codec_feature = reconstruction.features
        if lr_yuv.ndim != 5 or lr_yuv.shape[1] != t - 1:
            raise ValueError("MLVC reconstruct must return Bx(T-1)x3xHxW P-frames")
        hr_p = hr_rgb[:, 1:]
        if self.colorspace == "rgb":
            lr = _map_time(lr_yuv, mlvc_ycbcr_to_rgb)
            hr = hr_p
        else:
            lr = lr_yuv
            hr = _map_time(hr_p, rgb_to_mlvc_ycbcr)
        lr = lr.mul(255.0)
        hr = hr.mul(255.0)
        if training and self.codec_dropout > 0.0:
            disabled = (
                torch.rand(
                    codec_feature.shape[0],
                    codec_feature.shape[1],
                    1,
                    1,
                    1,
                    device=codec_feature.device,
                )
                < self.codec_dropout
            )
            codec_feature = torch.where(disabled, torch.zeros_like(codec_feature), codec_feature)
        if training:
            current = lr.flatten(0, 1).contiguous()
            target = hr.flatten(0, 1).contiguous()
            if not self.codec_context:
                return current, target
            codec = codec_feature.flatten(0, 1).contiguous()
            return (current, codec), target
        current = lr[:, -1].contiguous()
        target = hr[:, -1].contiguous()
        if not self.codec_context:
            return current, target
        return (current, codec_feature[:, -1].contiguous()), target


@dataclass
class MLVCDeviceBatch:
    lr: SRInput
    hr: torch.Tensor
    ready_event: torch.cuda.Event | None = None

    def __iter__(self) -> Iterator[object]:
        yield self.lr
        yield self.hr

    def wait_ready(self, stream: torch.cuda.Stream | None = None) -> None:
        if self.ready_event is None:
            return
        current = self.lr if isinstance(self.lr, torch.Tensor) else self.lr[0]
        consumer = stream or torch.cuda.current_stream(current.device)
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
        current = lr if isinstance(lr, torch.Tensor) else lr[0]
        if current.device.type == "cuda":
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(current.device))
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

    def __iter__(self) -> Iterator[tuple[SRInput, torch.Tensor]]:
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
    scale: int,
    colorspace: str,
    train_aug: bool,
    val_batch_size: int,
    rank: int,
    world_size: int,
    project_root: Path,
) -> tuple[MLVCTrainLoader, MLVCValidationLoader]:
    description = _resolve(data.dataset_description, project_root)
    video_root = _resolve(data.video_root, project_root) if data.video_root else description.parent
    sequences = load_openvid_index(description, video_root=video_root)
    train_indices, val_indices = split_sequence_indices(
        len(sequences), val_fraction=data.val_fraction, seed=data.split_seed
    )
    train_dataset = OpenVidSequenceDataset(
        sequences,
        indices=train_indices,
        sequence_frames=data.sequence_frames,
        lr_size=data.lr_size,
        hr_size=data.hr_size,
        training_split=True,
        augment=train_aug,
    )
    val_base_count = max(1, data.val_samples // len(data.q_indices))
    val_dataset = OpenVidSequenceDataset(
        sequences,
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
    processor = MLVCBatchProcessor(
        runtime,
        decoder=decoder,
        device=device,
        q_indices=data.q_indices,
        colorspace=colorspace,
        scale=scale,
        codec_context=data.codec_context,
        codec_dropout=data.codec_dropout,
    )
    train_iterator = _InfiniteMLVCIterator(train_cpu_loader, processor, train_sampler)
    return (
        MLVCTrainLoader(train_iterator, decoder),
        MLVCValidationLoader(val_cpu_loader, processor),
    )
