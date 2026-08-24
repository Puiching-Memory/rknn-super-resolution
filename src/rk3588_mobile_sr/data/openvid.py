"""OpenVidHD sequence index from MLVC ``frame_sequences.csv`` of source videos."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from rk3588_mobile_sr.data.decode import VIDEO_SUFFIXES, require_video_file
from rk3588_mobile_sr.utils.run_logger import logger


@dataclass(frozen=True)
class OpenVidSequence:
    path: str
    n_frames: int
    start_frame: int
    width: int
    height: int
    bbox: tuple[int, int, int, int]


def index_video_files(root: Path) -> dict[str, Path]:
    """Map video basenames under ``root`` to resolved paths (first match wins)."""
    catalog: dict[str, Path] = {}
    for suffix in sorted(VIDEO_SUFFIXES):
        for path in sorted(root.rglob(f"*{suffix}")):
            if path.is_file() and path.name not in catalog:
                catalog[path.name] = path.resolve()
    return catalog


def load_openvid_frame_sequences(
    path: str | Path,
    video_root: str | Path,
) -> list[OpenVidSequence]:
    """Load MLVC ``frame_sequences.csv`` and keep rows whose video exists under ``video_root``."""
    csv_path = Path(path)
    root = Path(video_root)
    catalog = index_video_files(root)
    required = {
        "filename",
        "start_frame",
        "n_frames",
        "width",
        "height",
        "bbox_top",
        "bbox_bottom",
        "bbox_left",
        "bbox_right",
    }
    sequences: list[OpenVidSequence] = []
    total = 0
    skipped_missing = 0
    skipped_scale = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"invalid OpenVidHD frame-sequence CSV: {csv_path}")
        for row in reader:
            total += 1
            if int(row.get("scale_factor", "1")) != 1:
                skipped_scale += 1
                continue
            source = catalog.get(row["filename"])
            if source is None:
                skipped_missing += 1
                continue
            n_frames = int(row["n_frames"])
            left = int(row["bbox_left"])
            top = int(row["bbox_top"])
            right = int(row["bbox_right"])
            bottom = int(row["bbox_bottom"])
            if n_frames < 2 or right <= left or bottom <= top:
                continue
            sequences.append(
                OpenVidSequence(
                    path=str(source),
                    n_frames=n_frames,
                    start_frame=int(row["start_frame"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    bbox=(left, top, right, bottom),
                )
            )
    if not sequences:
        raise ValueError(f"no OpenVidHD videos from {csv_path} found under {root}")
    logger.info(
        "OpenVidHD CSV {}: using {}/{} sequences (missing video={}, scale_factor!=1={}) under {}",
        csv_path,
        len(sequences),
        total,
        skipped_missing,
        skipped_scale,
        root,
    )
    return sequences


def load_openvid_index(
    path: str | Path,
    *,
    video_root: str | Path | None = None,
) -> list[OpenVidSequence]:
    """Load MLVC ``frame_sequences.csv`` and resolve videos under ``video_root``."""
    index = Path(path)
    if index.suffix.lower() != ".csv":
        raise ValueError(f"OpenVidHD index must be a .csv file, got {index}")
    root = Path(video_root) if video_root is not None else index.parent
    return load_openvid_frame_sequences(index, root)


def split_sequence_indices(
    count: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a stable source-level train/validation split."""
    if count < 2:
        raise ValueError("OpenVidHD training requires at least two independent sequences")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = min(count - 1, max(1, round(count * val_fraction)))
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def crop_box(width: int, height: int, *, training: bool) -> tuple[int, int, int, int]:
    target_ratio = 16.0 / 9.0
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_h = height
        crop_w = round(height * target_ratio)
        max_left = width - crop_w
        left = random.randint(0, max_left) if training and max_left else max_left // 2
        top = 0
    else:
        crop_w = width
        crop_h = round(width / target_ratio)
        max_top = height - crop_h
        top = random.randint(0, max_top) if training and max_top else max_top // 2
        left = 0
    return left, top, left + crop_w, top + crop_h


def collate_openvid_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate clip metadata for GPU-side video decode."""
    batch: dict[str, Any] = {
        "source": [sample["source"] for sample in samples],
        "frame_indices": torch.stack([sample["frame_indices"] for sample in samples]),
        "crop": torch.stack([sample["crop"] for sample in samples]),
        "hflip": torch.stack([sample["hflip"] for sample in samples]),
    }
    if "q_index" in samples[0]:
        batch["q_index"] = torch.stack([sample["q_index"] for sample in samples])
    return batch


class OpenVidSequenceDataset(Dataset[dict[str, Any]]):
    """Sample OpenVidHD clip metadata for GPU-side TorchCodec decoding."""

    def __init__(
        self,
        sequences: list[OpenVidSequence],
        *,
        indices: list[int],
        sequence_frames: int,
        lr_size: tuple[int, int],
        hr_size: tuple[int, int],
        training_split: bool,
        augment: bool,
        q_indices: tuple[int, ...] = (),
        max_samples: int | None = None,
    ) -> None:
        self.sequences = [sequences[index] for index in indices]
        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]
        if not self.sequences:
            raise ValueError("OpenVidHD split contains no sequences")
        if sequence_frames < 2:
            raise ValueError("sequence_frames must be at least 2 for MLVC P-frame reconstruction")
        if any(sequence.n_frames < sequence_frames for sequence in self.sequences):
            raise ValueError("every selected OpenVidHD sequence must cover sequence_frames")
        if not training_split and not q_indices:
            raise ValueError("validation requires at least one fixed q_index")

        self.sequence_frames = sequence_frames
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.training_split = training_split
        self.augment = augment
        self.q_indices = q_indices

    def __len__(self) -> int:
        multiplier = 1 if self.training_split else len(self.q_indices)
        return len(self.sequences) * multiplier

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.training_split:
            sequence = self.sequences[index % len(self.sequences)]
            q_index = None
        else:
            sequence_index, q_offset = divmod(index, len(self.q_indices))
            sequence = self.sequences[sequence_index]
            q_index = self.q_indices[q_offset]

        max_start = sequence.n_frames - self.sequence_frames
        start = random.randint(0, max_start) if self.augment and max_start else max_start // 2
        local_indices = list(range(start, start + self.sequence_frames))
        if self.augment and random.random() < 0.5:
            local_indices.reverse()

        source = require_video_file(sequence.path)
        left, top, right, bottom = sequence.bbox
        inner_left, inner_top, inner_right, inner_bottom = crop_box(
            right - left,
            bottom - top,
            training=self.augment,
        )
        sample: dict[str, Any] = {
            "source": str(source),
            "frame_indices": torch.tensor(
                [sequence.start_frame + offset for offset in local_indices],
                dtype=torch.long,
            ),
            "crop": torch.tensor(
                (
                    left + inner_left,
                    top + inner_top,
                    left + inner_right,
                    top + inner_bottom,
                ),
                dtype=torch.long,
            ),
            "hflip": torch.tensor(self.augment and random.random() < 0.5),
        }
        if q_index is not None:
            sample["q_index"] = torch.tensor(q_index, dtype=torch.long)
        return sample
