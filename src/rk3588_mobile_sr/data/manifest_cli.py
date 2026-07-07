"""CLI entry points for manifest generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

YUV_RE = re.compile(r"^(?P<name>.+)_1920x1080_(?P<fps>\d+)fps\.yuv$")
FRAME_BYTES = 1920 * 1080 * 3 // 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _uvg_entries(root: Path) -> list[dict]:
    yuv_dir = root / "data" / "UVG_raw" / "yuv_1080p"
    entries: list[dict] = []
    for path in sorted(yuv_dir.glob("*.yuv")):
        match = YUV_RE.match(path.name)
        if not match:
            continue
        frames = path.stat().st_size // FRAME_BYTES
        entries.append(
            {
                "id": f"uvg/{match.group('name')}",
                "type": "yuv_video",
                "path": path.relative_to(root).as_posix(),
                "width": 1920,
                "height": 1080,
                "fps": int(match.group("fps")),
                "frames": frames,
                "pix_fmt": "yuv420p",
                "bit_depth": 8,
                "weight": 1.0,
                "tags": ["uvg"],
            }
        )
    return entries


def _div2k_entries(root: Path, *, weight: float = 0.3) -> list[dict]:
    hr_dir = root / "data" / "DIV2K_train_HR"
    if not hr_dir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(hr_dir.glob("*.png")):
        rows.append(
            {
                "id": f"div2k/{path.stem}",
                "type": "image",
                "path": path.relative_to(root).as_posix(),
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "frames": 1,
                "weight": weight,
                "tags": ["texture", "div2k"],
            }
        )
    return rows


def build_train_manifest() -> None:
    root = _project_root()
    out = root / "data" / "sources" / "manifests" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = _uvg_entries(root) + _div2k_entries(root)
    with out.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} entries -> {out}")


def build_val_codec_fixed() -> None:
    root = _project_root()
    yuv_dir = root / "data" / "UVG_raw" / "yuv_1080p"
    out = root / "data" / "sources" / "manifests" / "val_fixed.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted(yuv_dir.glob("*.yuv"))[:4]:
        match = YUV_RE.match(path.name)
        if not match:
            continue
        frames = path.stat().st_size // FRAME_BYTES
        clip_start = min(60, max(0, frames - 48))
        frame_index = clip_start + 12
        base = {
            "id": f"uvg/{match.group('name')}",
            "type": "yuv_video",
            "path": path.relative_to(root).as_posix(),
            "width": 1920,
            "height": 1080,
            "fps": int(match.group("fps")),
            "frames": frames,
            "weight": 1.0,
            "clip_start": clip_start,
            "frame_index": frame_index,
            "encode_mode": "temporal_gop",
        }
        for codec in ("libx264", "libx265", "libsvtav1"):
            for bitrate in (150, 200, 300, 500, 800):
                row = dict(base)
                row["codec"] = codec
                row["bitrate_kbps"] = bitrate
                row["id"] = f"{base['id']}@{codec}@{bitrate}k"
                rows.append(row)

    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} val rows -> {out}")
