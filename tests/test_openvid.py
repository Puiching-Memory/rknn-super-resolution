"""Tests for MLVC-compatible OpenVidHD sequence loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rk3588_mobile_sr.data.openvid import (
    OpenVidSequenceDataset,
    load_openvid_description,
    split_sequence_indices,
)


def _write_dataset(root: Path, *, sequences: int = 3, frames: int = 4) -> Path:
    rows = []
    for sequence_index in range(sequences):
        sequence = root / f"sequence_{sequence_index:02d}"
        sequence.mkdir(parents=True)
        names = []
        for frame_index in range(frames):
            name = f"im{frame_index:05d}.png"
            image = np.full(
                (54, 96, 3),
                (sequence_index * 40 + frame_index * 5) % 255,
                dtype=np.uint8,
            )
            Image.fromarray(image).save(sequence / name)
            names.append(name)
        rows.append(
            {
                "path": sequence.name,
                "frames": names,
                "seq_length": frames,
                "width": 96,
                "height": 54,
            }
        )
    description = root / "description.json"
    description.write_text(json.dumps(rows), encoding="utf-8")
    return description


def test_load_openvid_description(tmp_path: Path):
    description = _write_dataset(tmp_path)
    sequences = load_openvid_description(description)
    assert len(sequences) == 3
    assert sequences[0].frames == tuple(f"im{i:05d}.png" for i in range(4))
    assert (sequences[0].height, sequences[0].width) == (54, 96)


def test_split_sequence_indices_is_stable_and_disjoint():
    first = split_sequence_indices(100, val_fraction=0.1, seed=42)
    second = split_sequence_indices(100, val_fraction=0.1, seed=42)
    train_indices, val_indices = first
    assert first == second
    assert len(train_indices) == 90
    assert len(val_indices) == 10
    assert set(train_indices).isdisjoint(val_indices)


def test_openvid_validation_sequence_shapes_and_q_indices(tmp_path: Path):
    description = _write_dataset(tmp_path)
    dataset = OpenVidSequenceDataset(
        description,
        indices=[0, 1],
        sequence_frames=3,
        lr_size=(12, 20),
        hr_size=(36, 60),
        training_split=False,
        augment=False,
        q_indices=(0, 63),
    )
    assert len(dataset) == 4
    sample = dataset[3]
    assert sample["lr_sequence"].shape == (3, 3, 12, 20)
    assert sample["hr"].shape == (3, 36, 60)
    assert sample["q_index"].item() == 63

