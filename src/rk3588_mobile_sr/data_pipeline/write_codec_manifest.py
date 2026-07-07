"""Write codec_cache/manifest.jsonl from planned jobs (no pixel reads)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.clip_plan import iter_encode_jobs, load_train_sources
from rk3588_mobile_sr.data_pipeline.paths import hr_mezzanine_path, lr_mp4_path
from rk3588_mobile_sr.data_pipeline.schemas import CodecClipRow


def build_codec_manifest_rows(
    *,
    root: Path,
    train_manifest: Path,
    cache_dir: Path,
    mezzanine_dir: Path,
    clip_frames: int,
    clips_per_video: int,
    codecs: list[str],
    bitrates_kbps: list[int],
    gop: int,
    image_gop: int,
    seed: int,
    lr_height: int,
    lr_width: int,
) -> list[CodecClipRow]:
    sources = load_train_sources(train_manifest)
    source_by_id = {s.id: s for s in sources}
    jobs = iter_encode_jobs(
        sources,
        clip_frames=clip_frames,
        clips_per_video=clips_per_video,
        codecs=codecs,
        bitrates_kbps=bitrates_kbps,
        gop=gop,
        image_gop=image_gop,
        seed=seed,
    )
    rows: list[CodecClipRow] = []
    for job in jobs:
        source = source_by_id[job["source_id"]]
        lr_abs = lr_mp4_path(
            cache_dir,
            source_id=source.id,
            clip_start=job["clip_start"],
            codec=job["codec"],
            bitrate_kbps=job["bitrate"],
            gop=job["gop"],
        )
        hr_abs = hr_mezzanine_path(mezzanine_dir, source.id)
        rows.append(
            CodecClipRow(
                id=f"{source.id}@s{job['clip_start']}@{job['codec']}@{job['bitrate']}k",
                path=lr_abs.relative_to(root).as_posix(),
                weight=source.weight,
                width=source.width,
                height=source.height,
                fps=job["fps"],
                frames=job["clip_frames"],
                source_id=source.id,
                source_path=source.path,
                clip_start=job["clip_start"],
                clip_frames=job["clip_frames"],
                codec=job["codec"],
                bitrate_kbps=job["bitrate"],
                gop=job["gop"],
                lr_height=lr_height,
                lr_width=lr_width,
                encode_mode=job["encode_mode"],
                hr_mp4_path=hr_abs.relative_to(root).as_posix(),
            )
        )
    return rows


def write_codec_manifest(
    *,
    root: Path,
    output_path: Path,
    train_manifest: Path,
    cache_dir: Path,
    mezzanine_dir: Path,
    clip_frames: int,
    clips_per_video: int,
    codecs: list[str],
    bitrates_kbps: list[int],
    gop: int,
    image_gop: int,
    seed: int,
    lr_height: int,
    lr_width: int,
) -> int:
    rows = build_codec_manifest_rows(
        root=root,
        train_manifest=train_manifest,
        cache_dir=cache_dir,
        mezzanine_dir=mezzanine_dir,
        clip_frames=clip_frames,
        clips_per_video=clips_per_video,
        codecs=codecs,
        bitrates_kbps=bitrates_kbps,
        gop=gop,
        image_gop=image_gop,
        seed=seed,
        lr_height=lr_height,
        lr_width=lr_width,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write codec cache manifest JSONL")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--mezzanine-dir", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--clips-per-video", type=int, default=8)
    parser.add_argument("--codecs", nargs="+", default=["libx264", "libx265", "libsvtav1"])
    parser.add_argument("--bitrates", nargs="+", type=int, default=[150, 200, 300, 500, 800])
    parser.add_argument("--gop", type=int, default=16)
    parser.add_argument("--image-gop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-height", type=int, default=360)
    parser.add_argument("--lr-width", type=int, default=640)
    args = parser.parse_args()
    root = args.root.resolve()
    n = write_codec_manifest(
        root=root,
        output_path=args.output.resolve(),
        train_manifest=(root / args.train_manifest).resolve()
        if not args.train_manifest.is_absolute()
        else args.train_manifest,
        cache_dir=(root / args.cache_dir).resolve()
        if not args.cache_dir.is_absolute()
        else args.cache_dir,
        mezzanine_dir=(root / args.mezzanine_dir).resolve()
        if not args.mezzanine_dir.is_absolute()
        else args.mezzanine_dir,
        clip_frames=args.clip_frames,
        clips_per_video=args.clips_per_video,
        codecs=args.codecs,
        bitrates_kbps=args.bitrates,
        gop=args.gop,
        image_gop=args.image_gop,
        seed=args.seed,
        lr_height=args.lr_height,
        lr_width=args.lr_width,
    )
    print(f"wrote {n} codec_clip entries -> {args.output}")


if __name__ == "__main__":
    main()
