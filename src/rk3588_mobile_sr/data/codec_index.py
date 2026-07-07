"""Expand codec_clip manifest rows into paired training frame index."""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rk3588_mobile_sr.data.manifest import load_manifest
from rk3588_mobile_sr.data.types import SourceRecord


@dataclass(frozen=True)
class CodecFrameEntry:
    """One aligned LR/HR canvas frame sample."""

    lr_path: Path
    hr_path: Path
    lr_frame: int
    hr_frame: int
    weight: float
    record_id: str


def expand_codec_clip_frames(
    records: list[SourceRecord],
    project_root: Path,
) -> list[CodecFrameEntry]:
    """Flatten codec_clip manifest rows to per-frame LR/HR decode entries."""
    entries: list[CodecFrameEntry] = []
    for record in records:
        if record.type != "codec_clip":
            continue
        hr_mp4 = record.extra.get("hr_mp4_path")
        if not hr_mp4:
            raise ValueError(
                f"codec_clip row {record.id!r} requires hr_mp4_path for canvas training"
            )
        lr_path = (project_root / record.path).resolve()
        hr_path = (project_root / hr_mp4).resolve()
        if not lr_path.is_file():
            raise FileNotFoundError(lr_path)
        if not hr_path.is_file():
            raise FileNotFoundError(hr_path)

        clip_start = int(record.extra.get("clip_start", 0))
        clip_frames = int(record.extra.get("clip_frames", record.frames))
        for rel in range(clip_frames):
            entries.append(
                CodecFrameEntry(
                    lr_path=lr_path,
                    hr_path=hr_path,
                    lr_frame=rel,
                    hr_frame=clip_start + rel,
                    weight=record.weight,
                    record_id=record.id,
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


def write_paired_file_lists(
    entries: list[CodecFrameEntry],
    *,
    lr_list_path: Path,
    hr_list_path: Path,
) -> None:
    """Write synchronized DALI file_list files (one frame range per line)."""
    lr_lines: list[str] = []
    hr_lines: list[str] = []
    for idx, entry in enumerate(entries):
        lr_lines.append(f"{entry.lr_path} {idx} {entry.lr_frame} {entry.lr_frame + 1}")
        hr_lines.append(f"{entry.hr_path} {idx} {entry.hr_frame} {entry.hr_frame + 1}")
    lr_list_path.write_text("\n".join(lr_lines) + "\n", encoding="utf-8")
    hr_list_path.write_text("\n".join(hr_lines) + "\n", encoding="utf-8")


@dataclass
class CodecFrameIndex:
    """Shuffled flat frame list plus optional DALI file lists."""

    entries: list[CodecFrameEntry]
    lr_list: Path | None = None
    hr_list: Path | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None


def build_codec_frame_index(
    manifest_path: str | Path,
    *,
    project_root: Path,
    seed: int = 0,
    weight_scale: int = 100,
    for_dali: bool = False,
) -> CodecFrameIndex:
    """Build a shuffled per-frame index from a codec cache manifest."""
    records = load_manifest(manifest_path, project_root=project_root)
    entries = expand_codec_clip_frames(records, project_root)
    rng = random.Random(seed)
    shuffled = _weighted_shuffle(entries, rng=rng, weight_scale=weight_scale)

    if not for_dali:
        return CodecFrameIndex(entries=shuffled)

    temp_dir = tempfile.TemporaryDirectory(prefix="rk3588_codec_")
    root = Path(temp_dir.name)
    lr_list = root / "lr.list"
    hr_list = root / "hr.list"
    write_paired_file_lists(shuffled, lr_list_path=lr_list, hr_list_path=hr_list)
    return CodecFrameIndex(
        entries=shuffled,
        lr_list=lr_list,
        hr_list=hr_list,
        temp_dir=temp_dir,
    )
