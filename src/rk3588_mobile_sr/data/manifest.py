"""JSONL manifest loading and validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from rk3588_mobile_sr.data.types import SourceRecord, ValSampleSpec


def load_manifest(path: str | Path, *, project_root: Path | None = None) -> list[SourceRecord]:
    """Load and validate a JSONL manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[SourceRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record = SourceRecord.from_dict(row)
            _validate_record(record, project_root=project_root)
            records.append(record)

    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def _validate_record(record: SourceRecord, *, project_root: Path | None) -> None:
    if record.type not in ("yuv_video", "codec_clip", "image"):
        raise ValueError(f"Unsupported source type {record.type!r} for {record.id}")
    if record.weight <= 0:
        raise ValueError(f"weight must be > 0 for {record.id}")

    if project_root is not None:
        resolved = (project_root / record.path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Source file missing for {record.id}: {resolved}")
        if record.type == "codec_clip":
            source_path = record.extra.get("source_path")
            if not source_path:
                raise ValueError(f"codec_clip {record.id} requires source_path")
            hr_mp4 = record.extra.get("hr_mp4_path")
            if hr_mp4:
                mp4_path = (project_root / hr_mp4).resolve()
                if not mp4_path.is_file():
                    raise FileNotFoundError(f"HR mezzanine missing for {record.id}: {mp4_path}")
            else:
                hr_path = (project_root / source_path).resolve()
                if not hr_path.is_file():
                    raise FileNotFoundError(f"HR source missing for {record.id}: {hr_path}")

    if record.type == "yuv_video":
        if record.width <= 0 or record.height <= 0 or record.frames <= 0:
            raise ValueError(f"yuv_video record {record.id} requires width/height/frames")
    if record.type == "image":
        if record.width <= 0 or record.height <= 0:
            raise ValueError(f"image record {record.id} requires width/height")


def weighted_choice(records: list[SourceRecord], rng: random.Random) -> SourceRecord:
    weights = [r.weight for r in records]
    return rng.choices(records, weights=weights, k=1)[0]


def load_val_manifest(path: str | Path, *, project_root: Path | None = None) -> list[ValSampleSpec]:
    """Load fixed validation manifest rows."""
    manifest_path = Path(path)
    specs: list[ValSampleSpec] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = SourceRecord.from_dict(row)
            _validate_record(source, project_root=project_root)
            specs.append(
                ValSampleSpec(
                    source=source,
                    frame_index=int(row.get("frame_index", 0)),
                    clip_start=int(row.get("clip_start", row.get("frame_index", 0))),
                    codec=str(row.get("codec", "libx264")),
                    bitrate_kbps=row.get("bitrate_kbps"),
                    crf=row.get("crf"),
                    encode_mode=row.get("encode_mode", "bitrate_sweep"),  # type: ignore[arg-type]
                )
            )
    if not specs:
        raise ValueError(f"Val manifest is empty: {manifest_path}")
    return specs
