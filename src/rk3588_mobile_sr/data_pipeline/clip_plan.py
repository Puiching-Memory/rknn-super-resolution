"""Clip start planning for the offline codec cache pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.schemas import SourceRow


def load_train_sources(manifest_path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(SourceRow.from_json(json.loads(line)))
    return rows


def clip_starts_for_record(
    source: SourceRow,
    *,
    clip_frames: int,
    clips_per_video: int,
    rng: random.Random,
) -> list[int]:
    """Deterministic clip start indices for offline codec cache planning."""
    if source.type == "image":
        return list(range(max(1, clips_per_video)))
    max_start = max(0, source.frames - clip_frames)
    if max_start == 0:
        return [0]
    return sorted(
        {rng.randint(0, max_start) for _ in range(min(clips_per_video, max_start + 1))}
    )


def iter_encode_jobs(
    sources: list[SourceRow],
    *,
    clip_frames: int,
    clips_per_video: int,
    codecs: list[str],
    bitrates_kbps: list[int],
    gop: int,
    image_gop: int,
    seed: int,
) -> list[dict]:
    """Enumerate LR encode targets as plain dicts for Snakefile expand()."""
    jobs: list[dict] = []
    rng = random.Random(seed)
    for source in sources:
        starts = clip_starts_for_record(
            source,
            clip_frames=clip_frames,
            clips_per_video=clips_per_video,
            rng=rng,
        )
        if source.type == "image":
            clip_n = 1
            job_gop = image_gop
            encode_mode = "intra_only"
        else:
            clip_n = clip_frames
            job_gop = gop
            encode_mode = "temporal_gop"
        for clip_start in starts:
            for codec in codecs:
                for bitrate in bitrates_kbps:
                    jobs.append(
                        {
                            "safe_id": source.safe_id(),
                            "source_id": source.id,
                            "clip_start": clip_start,
                            "clip_frames": clip_n,
                            "gop": job_gop,
                            "codec": codec,
                            "bitrate": bitrate,
                            "encode_mode": encode_mode,
                            "source_type": source.type,
                            "fps": source.fps or 30,
                        }
                    )
    return jobs


def unique_scale_jobs(jobs: list[dict]) -> list[dict]:
    """Deduplicate (safe_id, clip_start) pairs for scaled intermediate files."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for job in jobs:
        key = (job["safe_id"], job["clip_start"])
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out
