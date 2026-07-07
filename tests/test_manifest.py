"""Tests for manifest loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rk3588_mobile_sr.data.manifest import load_manifest, weighted_choice
from rk3588_mobile_sr.data.types import SourceRecord


def test_source_record_from_dict():
    row = {
        "id": "uvg/Beauty",
        "type": "yuv_video",
        "path": "data/foo.yuv",
        "width": 1920,
        "height": 1080,
        "frames": 10,
        "weight": 2.0,
    }
    rec = SourceRecord.from_dict(row)
    assert rec.id == "uvg/Beauty"
    assert rec.weight == 2.0


def test_load_manifest_train(tmp_path: Path):
    yuv = tmp_path / "clip.yuv"
    yuv.write_bytes(b"\x00" * (1920 * 1080 * 3 // 2))
    manifest = tmp_path / "train.jsonl"
    row = {
        "id": "uvg/Test",
        "type": "yuv_video",
        "path": yuv.name,
        "width": 1920,
        "height": 1080,
        "fps": 50,
        "frames": 1,
        "weight": 1.0,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = load_manifest(manifest, project_root=tmp_path)
    assert len(records) == 1
    assert records[0].id == "uvg/Test"


def test_weighted_choice_prefers_heavier():
    import random

    rng = random.Random(0)
    records = [
        SourceRecord.from_dict(
            {"id": "a", "type": "image", "path": "a.png", "weight": 1.0}
        ),
        SourceRecord.from_dict(
            {"id": "b", "type": "image", "path": "b.png", "weight": 100.0}
        ),
    ]
    picks = [weighted_choice(records, rng).id for _ in range(50)]
    assert picks.count("b") > picks.count("a")


def test_load_manifest_missing_file():
    with pytest.raises(FileNotFoundError):
        load_manifest("/nonexistent/train.jsonl")
