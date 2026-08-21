"""Tests for MLVC-compatible OpenVidHD sequence metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rk3588_mobile_sr.data.openvid import (
    OpenVidSequenceDataset,
    collate_openvid_batch,
    crop_box,
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


def test_openvid_validation_metadata_and_q_indices(tmp_path: Path):
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
    assert sample["kind"] == "images"
    assert len(sample["paths"]) == 3
    assert sample["frame_indices"].tolist() == [0, 1, 2]
    assert sample["crop"].tolist() == [0, 0, 96, 54]
    assert sample["hflip"].item() is False
    assert sample["q_index"].item() == 63
    assert Path(sample["paths"][0]).is_file()


def test_openvid_prefers_video_sidecar(tmp_path: Path):
    description = _write_dataset(tmp_path, sequences=2, frames=3)
    sidecar = tmp_path / "sequence_00" / "sequence.mp4"
    sidecar.write_bytes(b"placeholder")
    dataset = OpenVidSequenceDataset(
        description,
        indices=[0],
        sequence_frames=2,
        lr_size=(12, 20),
        hr_size=(36, 60),
        training_split=False,
        augment=False,
        q_indices=(21,),
    )
    sample = dataset[0]
    assert sample["kind"] == "video"
    assert sample["source"] == str(sidecar.resolve())
    assert sample["paths"] == []
    assert sample["frame_indices"].tolist() == [0, 1]


def test_collate_openvid_batch_keeps_path_lists(tmp_path: Path):
    description = _write_dataset(tmp_path, sequences=2, frames=3)
    dataset = OpenVidSequenceDataset(
        description,
        indices=[0, 1],
        sequence_frames=2,
        lr_size=(12, 20),
        hr_size=(36, 60),
        training_split=False,
        augment=False,
        q_indices=(0,),
    )
    batch = collate_openvid_batch([dataset[0], dataset[1]])
    assert batch["kind"] == ["images", "images"]
    assert len(batch["paths"]) == 2
    assert batch["frame_indices"].shape == (2, 2)
    assert batch["crop"].shape == (2, 4)
    assert batch["q_index"].tolist() == [0, 0]
    assert torch.equal(batch["hflip"], torch.tensor([False, False]))


def test_load_openvid_shared_frames_description(tmp_path: Path):
    frames = [f"im{i:05d}.png" for i in range(3)]
    sequence = tmp_path / "sequence_00"
    sequence.mkdir()
    for name in frames:
        Image.fromarray(np.zeros((54, 96, 3), dtype=np.uint8)).save(sequence / name)
    description = tmp_path / "description.json"
    description.write_text(
        json.dumps(
            {
                "frames": frames,
                "seqs": [
                    {
                        "path": sequence.name,
                        "seq_length": 3,
                        "width": 96,
                        "height": 54,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sequences = load_openvid_description(description)
    assert len(sequences) == 1
    assert sequences[0].frames == tuple(frames)
    assert (sequences[0].height, sequences[0].width) == (54, 96)


def test_crop_box_centers_16_9_when_not_training():
    left, top, right, bottom = crop_box(200, 100, training=False)
    assert top == 0
    assert bottom - top == 100
    assert right - left == round(100 * 16 / 9)
