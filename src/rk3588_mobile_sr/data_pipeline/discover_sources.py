"""Discover UVG YUV sources; write train/val manifests.

Still-image (DIV2K) support has been removed: the pipeline is now driven
entirely by native YUV420p video sequences so that the supervision signal and
the simulated capture/codec degradation chain match real video content.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.schemas import SourceRow, ValRow

YUV_RE = re.compile(r"^(?P<name>.+)_1920x1080_(?P<fps>\d+)fps\.yuv$")
FRAME_BYTES = 1920 * 1080 * 3 // 2


def uvg_entries(root: Path) -> list[SourceRow]:
    yuv_dir = root / "data" / "UVG_raw" / "yuv_1080p"
    entries: list[SourceRow] = []
    for path in sorted(yuv_dir.glob("*.yuv")):
        match = YUV_RE.match(path.name)
        if not match:
            continue
        frames = path.stat().st_size // FRAME_BYTES
        entries.append(
            SourceRow(
                id=f"uvg/{match.group('name')}",
                type="yuv_video",
                path=path.relative_to(root).as_posix(),
                width=1920,
                height=1080,
                fps=int(match.group("fps")),
                frames=frames,
                tags=["uvg"],
            )
        )
    return entries


def val_entries(root: Path) -> list[ValRow]:
    yuv_dir = root / "data" / "UVG_raw" / "yuv_1080p"
    rows: list[ValRow] = []
    for path in sorted(yuv_dir.glob("*.yuv"))[:4]:
        match = YUV_RE.match(path.name)
        if not match:
            continue
        frames = path.stat().st_size // FRAME_BYTES
        clip_start = min(60, max(0, frames - 48))
        frame_index = clip_start + 12
        base_id = f"uvg/{match.group('name')}"
        base_path = path.relative_to(root).as_posix()
        for codec in ("libx264", "libx265", "libsvtav1"):
            for bitrate in (150, 200, 300, 500, 800):
                rows.append(
                    ValRow(
                        id=f"{base_id}@{codec}@{bitrate}k",
                        path=base_path,
                        fps=int(match.group("fps")),
                        frames=frames,
                        clip_start=clip_start,
                        frame_index=frame_index,
                        codec=codec,
                        bitrate_kbps=bitrate,
                    )
                )
    return rows


def write_train_manifest(root: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = uvg_entries(root)
    with out.open("w", encoding="utf-8") as handle:
        for row in entries:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
    return len(entries)


def write_val_manifest(root: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = val_entries(root)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover training/val source manifests")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/sources/manifests/train.jsonl"),
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("data/sources/manifests/val_fixed.jsonl"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    train_out = (
        (root / args.train_out).resolve()
        if not args.train_out.is_absolute()
        else args.train_out
    )
    val_out = (
        (root / args.val_out).resolve() if not args.val_out.is_absolute() else args.val_out
    )
    n_train = write_train_manifest(root, train_out)
    n_val = write_val_manifest(root, val_out)
    print(f"wrote {n_train} train entries -> {train_out}")
    print(f"wrote {n_val} val entries -> {val_out}")


if __name__ == "__main__":
    main()
