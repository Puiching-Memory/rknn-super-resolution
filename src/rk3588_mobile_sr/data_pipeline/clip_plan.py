"""Clip start planning and codec job sampling for the offline codec cache.

Still-image sources have been removed: every job is a temporal YUV clip. GOP
and bitrate are now *sampled* from realistic distributions (log-normal bitrate,
weighted GOP candidates) instead of a uniform cartesian product, so the cache
matches the bitrate/GOP skew of real streamed video.

Fixed validation rows (``val_fixed.jsonl``) additionally get dedicated encode
jobs so every ``(sequence, codec, bitrate)`` canvas is covered at the val
``clip_start`` / ``frame_index`` — train sampling alone does not guarantee that.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.schemas import SourceRow, ValRow

# Real-world video bitrate is roughly log-normal with a long tail toward low
# rates. Per-codec center (kbps) reflects that x265/AV1 deliver comparable
# quality at lower rate than x264.
_CODEC_BITRATE_CENTER = {"libx264": 300.0, "libx265": 220.0, "libsvtav1": 200.0}
_BITRATE_LOG_SIGMA = 0.35

# Default GOP candidates (frames) and their empirical weights: short GOPs
# dominate live/conferencing, longer GOPs are typical for VoD streaming.
DEFAULT_GOP_CANDIDATES = [30, 60, 120]
DEFAULT_GOP_WEIGHTS = [0.4, 0.4, 0.2]
# Deterministic GOP for fixed-val bake jobs (must cover val frame_index).
DEFAULT_VAL_GOP = 30


def load_train_sources(manifest_path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(SourceRow.from_json(json.loads(line)))
    return rows


def load_val_rows(manifest_path: Path) -> list[ValRow]:
    rows: list[ValRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(ValRow.model_validate(json.loads(line)))
    return rows


def clip_starts_for_record(
    source: SourceRow,
    *,
    clip_frames: int,
    clips_per_video: int,
    rng: random.Random,
) -> list[int]:
    """Deterministic, temporally-stratified clip start indices.

    The video span is split into ``clips_per_video`` strata and one start is
    sampled per stratum, so clips cover the whole timeline instead of
    clustering (the previous set-dedup randint approach tended to bunch up).
    """
    max_start = max(0, source.frames - clip_frames)
    if max_start == 0:
        return [0]
    n = min(clips_per_video, max_start + 1)
    step = max_start / n
    starts: set[int] = set()
    for i in range(n):
        lo = i * step
        hi = (i + 1) * step
        starts.add(min(max_start, int(rng.uniform(lo, hi))))
    return sorted(starts)


def _sample_bitrate(
    rng: random.Random,
    candidates: list[int],
    *,
    codec: str,
) -> int:
    """Sample a bitrate (kbps) from a log-normal centred on the codec, snapped
    to the nearest candidate tier. Lower/mid tiers are favoured."""
    if not candidates:
        raise ValueError("bitrates_kbps must be non-empty")
    if len(candidates) == 1:
        return candidates[0]
    center = _CODEC_BITRATE_CENTER.get(codec, 250.0)
    log_target = rng.gauss(math.log(center), _BITRATE_LOG_SIGMA)
    return min(candidates, key=lambda b: abs(math.log(b) - log_target))


def _sample_gop(
    rng: random.Random,
    candidates: list[int],
    weights: list[float] | None,
) -> int:
    if not candidates:
        raise ValueError("gop_candidates must be non-empty")
    if len(candidates) == 1:
        return candidates[0]
    w = weights if weights else [1.0] * len(candidates)
    if len(w) != len(candidates):
        raise ValueError("gop_weights length must match gop_candidates")
    return rng.choices(candidates, weights=w, k=1)[0]


def _job_key(job: dict) -> tuple:
    return (
        job["safe_id"],
        job["clip_start"],
        job["gop"],
        job["codec"],
        job["bitrate"],
    )


def iter_encode_jobs(
    sources: list[SourceRow],
    *,
    clip_frames: int,
    clips_per_video: int,
    codecs: list[str],
    bitrates_kbps: list[int],
    gop_candidates: list[int],
    gop_weights: list[float] | None,
    seed: int,
) -> list[dict]:
    """Enumerate LR encode targets as plain dicts for Snakefile expand().

    Each (source, clip, codec) yields exactly one job whose bitrate and GOP are
    sampled from realistic distributions, replacing the old uniform cartesian
    product over all bitrate tiers.
    """
    jobs: list[dict] = []
    rng = random.Random(seed)
    for source in sources:
        starts = clip_starts_for_record(
            source,
            clip_frames=clip_frames,
            clips_per_video=clips_per_video,
            rng=rng,
        )
        for clip_start in starts:
            for codec in codecs:
                bitrate = _sample_bitrate(rng, bitrates_kbps, codec=codec)
                gop = _sample_gop(rng, gop_candidates, gop_weights)
                jobs.append(
                    {
                        "safe_id": source.safe_id(),
                        "source_id": source.id,
                        "clip_start": clip_start,
                        "clip_frames": clip_frames,
                        "gop": gop,
                        "codec": codec,
                        "bitrate": bitrate,
                        "encode_mode": "temporal_gop",
                        "source_type": "yuv_video",
                        "fps": source.fps or 30,
                        "role": "train",
                    }
                )
    return jobs


def iter_val_encode_jobs(
    val_rows: list[ValRow],
    sources: list[SourceRow],
    *,
    clip_frames: int,
    gop: int = DEFAULT_VAL_GOP,
) -> list[dict]:
    """Dedicated encode jobs covering every fixed-val ``(seq, codec, bitrate)``.

    Uses each val row's ``clip_start`` so ``frame_index`` falls inside the baked
    window (``clip_start <= frame_index < clip_start + clip_frames``).
    """
    source_by_id = {source.id: source for source in sources}
    jobs: list[dict] = []
    seen: set[tuple] = set()
    for row in val_rows:
        source_id = row.id.rsplit("@", 2)[0]
        source = source_by_id.get(source_id)
        if source is None:
            raise KeyError(
                f"val row {row.id!r} references unknown source {source_id!r}; "
                "rebuild train.jsonl / val_fixed.jsonl together"
            )
        if not (row.clip_start <= row.frame_index < row.clip_start + clip_frames):
            raise ValueError(
                f"val row {row.id!r}: frame_index={row.frame_index} outside "
                f"[{row.clip_start}, {row.clip_start + clip_frames})"
            )
        job = {
            "safe_id": source.safe_id(),
            "source_id": source.id,
            "clip_start": row.clip_start,
            "clip_frames": clip_frames,
            "gop": gop,
            "codec": row.codec,
            "bitrate": row.bitrate_kbps,
            "encode_mode": "temporal_gop",
            "source_type": "yuv_video",
            "fps": source.fps or row.fps or 30,
            "role": "val",
        }
        key = _job_key(job)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(job)
    return jobs


def merge_encode_jobs(*job_lists: list[dict]) -> list[dict]:
    """Concatenate encode job lists, dropping duplicate bake keys."""
    merged: list[dict] = []
    seen: set[tuple] = set()
    for jobs in job_lists:
        for job in jobs:
            key = _job_key(job)
            if key in seen:
                continue
            seen.add(key)
            merged.append(job)
    return merged
