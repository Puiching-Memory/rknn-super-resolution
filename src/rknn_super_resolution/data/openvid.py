"""OpenVidHD sequence index from MLVC ``frame_sequences.csv`` of source videos."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from rknn_super_resolution.data.decode import VIDEO_SUFFIXES, require_video_file
from rknn_super_resolution.utils.run_logger import logger


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
    sequences: Sequence[OpenVidSequence],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    manifest_path: str | Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Create or load a stable source-level train/validation/test split."""
    if val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("val_fraction and test_fraction must be positive")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    indices_by_source: dict[str, list[int]] = {}
    paths_by_source: dict[str, str] = {}
    for index, sequence in enumerate(sequences):
        source = Path(sequence.path).name
        existing_path = paths_by_source.setdefault(source, sequence.path)
        if existing_path != sequence.path:
            raise ValueError(f"source filename is not unique: {source}")
        indices_by_source.setdefault(source, []).append(index)

    sources = list(indices_by_source)
    if len(sources) < 3:
        raise ValueError("OpenVidHD training requires at least three independent source videos")

    split_sources: dict[str, list[str]]
    manifest = Path(manifest_path) if manifest_path is not None else None
    if manifest is not None and manifest.is_file():
        split_sources = _load_split_manifest(
            manifest,
            sources=sources,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        random.Random(seed).shuffle(sources)
        val_count = max(1, math.ceil(len(sources) * val_fraction))
        test_count = max(1, math.ceil(len(sources) * test_fraction))
        if val_count + test_count >= len(sources):
            raise ValueError("split fractions leave no source videos for training")
        split_sources = {
            "validation": sources[:val_count],
            "test": sources[val_count : val_count + test_count],
            "train": sources[val_count + test_count :],
        }
        if manifest is not None:
            _write_split_manifest(
                manifest,
                split_sources=split_sources,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )

    train_sources = split_sources["train"]
    val_sources = split_sources["validation"]
    test_sources = split_sources["test"]
    train_indices = [index for source in train_sources for index in indices_by_source[source]]
    val_indices = [index for source in val_sources for index in indices_by_source[source]]
    test_indices = [index for source in test_sources for index in indices_by_source[source]]
    return train_indices, val_indices, test_indices


def _load_split_manifest(
    path: Path,
    *,
    sources: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["version"] != 1:
            raise ValueError(f"unsupported version {payload['version']}")
        if payload["seed"] != seed:
            raise ValueError("split_seed does not match")
        if payload["val_fraction"] != val_fraction:
            raise ValueError("val_fraction does not match")
        if payload["test_fraction"] != test_fraction:
            raise ValueError("test_fraction does not match")
        raw_splits = payload["splits"]
        split_sources = {name: list(raw_splits[name]) for name in ("train", "validation", "test")}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid OpenVidHD split manifest: {path}") from exc

    if any(not isinstance(source, str) for split in split_sources.values() for source in split):
        raise ValueError(f"invalid OpenVidHD split manifest: {path}")
    if any(len(split) != len(set(split)) for split in split_sources.values()):
        raise ValueError(f"OpenVidHD split manifest contains duplicate source videos: {path}")
    source_sets = [set(split_sources[name]) for name in ("train", "validation", "test")]
    if any(not source_set for source_set in source_sets):
        raise ValueError(f"OpenVidHD split manifest contains an empty split: {path}")
    if any(source_sets[left] & source_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError(f"OpenVidHD split manifest contains overlapping source videos: {path}")
    manifest_sources = set().union(*source_sets)
    current_sources = set(sources)
    if manifest_sources != current_sources:
        missing = len(manifest_sources - current_sources)
        added = len(current_sources - manifest_sources)
        raise ValueError(
            "OpenVidHD sources differ from the fixed split manifest "
            f"(missing={missing}, added={added}): {path}"
        )
    return split_sources


def _write_split_manifest(
    path: Path,
    *,
    split_sources: dict[str, list[str]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    payload = {
        "version": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "splits": split_sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_unique_source_indices(
    sequences: Sequence[OpenVidSequence],
    indices: Sequence[int],
) -> list[int]:
    """Keep one representative sequence for each source video, preserving order."""
    selected: list[int] = []
    seen: set[str] = set()
    for index in indices:
        source = sequences[index].path
        if source not in seen:
            seen.add(source)
            selected.append(index)
    return selected


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
