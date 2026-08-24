"""Tests for OpenVidHD CSV clip metadata."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from rk3588_mobile_sr.data.openvid import (
    OpenVidSequenceDataset,
    collate_openvid_batch,
    crop_box,
    load_openvid_frame_sequences,
    load_openvid_index,
    split_sequence_indices,
)

_CSV_FIELDS = [
    "sequence_id",
    "filename",
    "start_frame",
    "n_frames",
    "scale_factor",
    "bbox_top",
    "bbox_bottom",
    "bbox_left",
    "bbox_right",
    "width",
    "height",
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    sequence_id: str,
    filename: str,
    *,
    start_frame: int = 0,
    n_frames: int = 64,
    scale_factor: int = 1,
    bbox: tuple[int, int, int, int] = (0, 0, 1920, 1080),
    width: int = 1920,
    height: int = 1080,
) -> dict[str, object]:
    left, top, right, bottom = bbox
    return {
        "sequence_id": sequence_id,
        "filename": filename,
        "start_frame": start_frame,
        "n_frames": n_frames,
        "scale_factor": scale_factor,
        "bbox_top": top,
        "bbox_bottom": bottom,
        "bbox_left": left,
        "bbox_right": right,
        "width": width,
        "height": height,
    }


def test_split_sequence_indices_is_stable_and_disjoint():
    first = split_sequence_indices(100, val_fraction=0.1, seed=42)
    second = split_sequence_indices(100, val_fraction=0.1, seed=42)
    train_indices, val_indices = first
    assert first == second
    assert len(train_indices) == 90
    assert len(val_indices) == 10
    assert set(train_indices).isdisjoint(val_indices)


def test_crop_box_centers_16_9_when_not_training():
    left, top, right, bottom = crop_box(200, 100, training=False)
    assert top == 0
    assert bottom - top == 100
    assert right - left == round(100 * 16 / 9)


def test_load_openvid_csv_skips_missing_video_and_uses_start_frame(tmp_path: Path):
    part = tmp_path / "part10"
    part.mkdir()
    (part / "keep_a.mp4").write_bytes(b"video")
    (part / "keep_b.mp4").write_bytes(b"video")
    csv_path = _write_csv(
        tmp_path / "frame_sequences.csv",
        [
            _row("000/000", "keep_a.mp4", start_frame=15),
            _row("000/001", "missing.mp4"),
            _row("000/002", "keep_b.mp4", start_frame=40, scale_factor=2),
            _row("000/003", "keep_b.mp4", start_frame=8, bbox=(20, 10, 1900, 1070)),
        ],
    )

    sequences = load_openvid_frame_sequences(csv_path, tmp_path)
    assert [item.path for item in sequences] == [
        str((part / "keep_a.mp4").resolve()),
        str((part / "keep_b.mp4").resolve()),
    ]
    assert sequences[0].start_frame == 15
    assert sequences[1].bbox == (20, 10, 1900, 1070)

    dataset = OpenVidSequenceDataset(
        sequences,
        indices=[0],
        sequence_frames=8,
        lr_size=(360, 640),
        hr_size=(1080, 1920),
        training_split=False,
        augment=False,
        q_indices=(21,),
    )
    sample = dataset[0]
    assert sample["source"] == str((part / "keep_a.mp4").resolve())
    assert "kind" not in sample
    assert "paths" not in sample
    assert sample["frame_indices"].tolist() == list(range(15 + 28, 15 + 28 + 8))
    assert sample["crop"].tolist() == [0, 0, 1920, 1080]
    assert sample["q_index"].item() == 21


def test_collate_openvid_batch_stacks_video_metadata(tmp_path: Path):
    part = tmp_path / "videos"
    part.mkdir()
    (part / "a.mp4").write_bytes(b"video")
    (part / "b.mp4").write_bytes(b"video")
    sequences = load_openvid_frame_sequences(
        _write_csv(
            tmp_path / "seqs.csv",
            [
                _row("000/000", "a.mp4", start_frame=4, n_frames=16, bbox=(0, 0, 96, 54), width=96, height=54),
                _row("000/001", "b.mp4", start_frame=2, n_frames=16, bbox=(0, 0, 96, 54), width=96, height=54),
            ],
        ),
        tmp_path,
    )
    dataset = OpenVidSequenceDataset(
        sequences,
        indices=[0, 1],
        sequence_frames=4,
        lr_size=(12, 20),
        hr_size=(36, 60),
        training_split=False,
        augment=False,
        q_indices=(0,),
    )
    batch = collate_openvid_batch([dataset[0], dataset[1]])
    assert len(batch["source"]) == 2
    assert batch["frame_indices"].shape == (2, 4)
    assert batch["crop"].shape == (2, 4)
    assert batch["q_index"].tolist() == [0, 0]
    assert torch.equal(batch["hflip"], torch.tensor([False, False]))
    assert dataset[0]["frame_indices"].tolist() == list(range(4 + 6, 4 + 6 + 4))


def test_load_openvid_index_requires_csv(tmp_path: Path):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    csv_path = _write_csv(
        tmp_path / "seqs.csv",
        [_row("000/000", "clip.mp4", n_frames=8, bbox=(0, 0, 96, 54), width=96, height=54)],
    )
    rows = load_openvid_index(csv_path, video_root=tmp_path)
    assert len(rows) == 1
    assert rows[0].n_frames == 8

    json_path = tmp_path / "description.json"
    json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.csv"):
        load_openvid_index(json_path, video_root=tmp_path)
