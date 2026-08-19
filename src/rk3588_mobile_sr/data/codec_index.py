"""Expand codec_clip manifest rows into paired training frame index."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from rk3588_mobile_sr.data.manifest import load_manifest
from rk3588_mobile_sr.data.types import SourceRecord


@dataclass(frozen=True)
class CodecFrameEntry:
    """One aligned LR/HR canvas frame sample.

    Both ``lr_path`` and ``hr_path`` are offline-baked RGB uint8 ``.npy`` clips.
    Frame indices are clip-relative (0 .. clip_frames-1).
    """

    lr_path: Path
    hr_path: Path
    lr_frame: int
    hr_frame: int
    weight: float
    record_id: str
    codec: str = ""


def expand_codec_clip_frames(
    records: list[SourceRecord],
    project_root: Path,
) -> list[CodecFrameEntry]:
    """Flatten codec_clip manifest rows to per-frame LR/HR decode entries."""
    entries: list[CodecFrameEntry] = []
    for record in records:
        if record.type != "codec_clip":
            continue
        hr_rel = record.extra.get("hr_path")
        if not hr_rel:
            raise ValueError(
                f"codec_clip row {record.id!r} requires hr_path (.npy) for training"
            )
        lr_path = (project_root / record.path).resolve()
        hr_path = (project_root / hr_rel).resolve()
        if not lr_path.is_file():
            raise FileNotFoundError(lr_path)
        if not hr_path.is_file():
            raise FileNotFoundError(hr_path)

        clip_frames = int(record.extra.get("clip_frames", record.frames))
        gop = int(record.extra.get("gop", 1))
        for rel in range(clip_frames):
            # I-frames (rel % gop == 0) are clean and easy to super-resolve;
            # P-frames carry the codec blocking/ringing we want the model to
            # see more of, so they get a higher sampling weight.
            is_intra = (rel % max(gop, 1)) == 0
            frame_weight = 0.5 if is_intra else 1.0
            entries.append(
                CodecFrameEntry(
                    lr_path=lr_path,
                    hr_path=hr_path,
                    lr_frame=rel,
                    hr_frame=rel,
                    weight=record.weight * frame_weight,
                    record_id=record.id,
                    codec=str(record.extra.get("codec", "")),
                )
            )
    if not entries:
        raise ValueError("No codec_clip frames found in manifest")
    return entries


def _weighted_shuffle(
    entries: list[CodecFrameEntry],
    *,
    rng: random.Random,
    weight_scale: int = 100,
) -> list[CodecFrameEntry]:
    expanded: list[CodecFrameEntry] = []
    for entry in entries:
        reps = max(1, int(round(entry.weight * weight_scale)))
        expanded.extend([entry] * reps)
    rng.shuffle(expanded)
    return expanded


@dataclass
class CodecFrameIndex:
    """Shuffled flat per-frame LR/HR .npy index."""

    entries: list[CodecFrameEntry]


def build_codec_frame_index(
    manifest_path: str | Path,
    *,
    project_root: Path,
    seed: int = 0,
    weight_scale: int = 100,
) -> CodecFrameIndex:
    """Build a shuffled per-frame index from a codec cache manifest."""
    records = load_manifest(manifest_path, project_root=project_root)
    entries = expand_codec_clip_frames(records, project_root)
    rng = random.Random(seed)
    shuffled = _weighted_shuffle(entries, rng=rng, weight_scale=weight_scale)
    return CodecFrameIndex(entries=shuffled)
