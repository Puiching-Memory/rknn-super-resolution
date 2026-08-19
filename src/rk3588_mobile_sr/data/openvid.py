"""OpenVidHD sequence dataset using MLVC's ``description.json`` format."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class OpenVidSequence:
    path: str
    frames: tuple[str, ...]
    width: int
    height: int


def load_openvid_description(path: str | Path) -> list[OpenVidSequence]:
    """Load both description layouts accepted by MLVC's FastVideoFolder."""
    description = Path(path)
    with description.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        rows = raw
        shared_frames = None
    elif isinstance(raw, dict) and "seqs" in raw and "frames" in raw:
        rows = raw["seqs"]
        shared_frames = raw["frames"]
    else:
        raise ValueError(f"invalid MLVC dataset description: {description}")

    sequences: list[OpenVidSequence] = []
    for index, row in enumerate(rows):
        frames = shared_frames if shared_frames is not None else row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"sequence {index} has no frames in {description}")
        seq_length = int(row.get("seq_length", len(frames)))
        if seq_length > len(frames):
            raise ValueError(
                f"sequence {index} declares {seq_length} frames but lists only {len(frames)}"
            )
        sequences.append(
            OpenVidSequence(
                path=str(row["path"]),
                frames=tuple(str(name) for name in frames[:seq_length]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    if not sequences:
        raise ValueError(f"empty MLVC dataset description: {description}")
    return sequences


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


def _crop_box(width: int, height: int, *, training: bool) -> tuple[int, int, int, int]:
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


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class OpenVidSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Read consecutive OpenVidHD frames and form 640x360 MLVC inputs."""

    def __init__(
        self,
        description: str | Path,
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
        self.description = Path(description).resolve()
        self.root = self.description.parent
        all_sequences = load_openvid_description(self.description)
        self.sequences = [all_sequences[index] for index in indices]
        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]
        if not self.sequences:
            raise ValueError("OpenVidHD split contains no sequences")
        if sequence_frames < 2:
            raise ValueError("sequence_frames must be at least 2 for MLVC P-frame reconstruction")
        if any(len(sequence.frames) < sequence_frames for sequence in self.sequences):
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.training_split:
            sequence = self.sequences[index % len(self.sequences)]
            q_index = None
        else:
            sequence_index, q_offset = divmod(index, len(self.q_indices))
            sequence = self.sequences[sequence_index]
            q_index = self.q_indices[q_offset]

        max_start = len(sequence.frames) - self.sequence_frames
        start = random.randint(0, max_start) if self.augment and max_start else max_start // 2
        names = list(sequence.frames[start : start + self.sequence_frames])
        if self.augment and random.random() < 0.5:
            names.reverse()

        first_path = self.root / sequence.path / names[0]
        with Image.open(first_path) as first:
            size = first.size
        crop = _crop_box(*size, training=self.augment)
        flip = self.augment and random.random() < 0.5
        lr_h, lr_w = self.lr_size
        hr_h, hr_w = self.hr_size

        lr_frames: list[torch.Tensor] = []
        hr_target: torch.Tensor | None = None
        for frame_offset, name in enumerate(names):
            frame_path = self.root / sequence.path / name
            with Image.open(frame_path) as source:
                image = source.convert("RGB").crop(crop)
                if flip:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                lr_image = image.resize((lr_w, lr_h), Image.Resampling.LANCZOS)
                lr_frames.append(_image_tensor(lr_image))
                if frame_offset == len(names) - 1:
                    hr_image = image.resize((hr_w, hr_h), Image.Resampling.LANCZOS)
                    hr_target = _image_tensor(hr_image)

        assert hr_target is not None
        sample = {
            "lr_sequence": torch.stack(lr_frames, dim=0),
            "hr": hr_target,
        }
        if q_index is not None:
            sample["q_index"] = torch.tensor(q_index, dtype=torch.long)
        return sample
