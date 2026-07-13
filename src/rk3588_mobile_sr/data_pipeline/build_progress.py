"""Periodic tqdm-style progress lines for the Snakemake data build."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rk3588_mobile_sr.data_pipeline.clip_plan import (
    DEFAULT_GOP_CANDIDATES,
    DEFAULT_GOP_WEIGHTS,
    iter_encode_jobs,
    load_train_sources,
)


@dataclass(frozen=True)
class BuildTargets:
    hr_clip: int
    codec: int

    @property
    def total(self) -> int:
        return self.hr_clip + self.codec


@dataclass(frozen=True)
class BuildCounts:
    hr_clip: int
    codec: int

    @property
    def total(self) -> int:
        return self.hr_clip + self.codec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_targets(root: Path, config_path: Path) -> BuildTargets:
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train_manifest = root / cfg["train_manifest"]
    sources = load_train_sources(train_manifest)
    jobs = iter_encode_jobs(
        sources,
        clip_frames=cfg["clip_frames"],
        clips_per_video=cfg["clips_per_video"],
        codecs=cfg["codecs"],
        bitrates_kbps=cfg["bitrates_kbps"],
        gop_candidates=cfg.get("gop_candidates", DEFAULT_GOP_CANDIDATES),
        gop_weights=cfg.get("gop_weights", DEFAULT_GOP_WEIGHTS),
        seed=cfg["seed"],
    )
    # Each HR lossless clip is shared across all codec jobs for that clip.
    hr_clip_n = len({(j["safe_id"], j["clip_start"]) for j in jobs})
    return BuildTargets(hr_clip=hr_clip_n, codec=len(jobs))


def _count_suffix_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix))


def count_outputs(root: Path, config_path: Path) -> BuildCounts:
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    hr_dir = root / cfg["hr_dir"]
    codec_dir = root / cfg["codec_cache_dir"]
    return BuildCounts(
        hr_clip=_count_suffix_files(hr_dir, "_hr.mp4"),
        codec=_count_suffix_files(codec_dir, ".mp4"),
    )


def _bar(ratio: float, width: int = 28) -> str:
    ratio = min(max(ratio, 0.0), 1.0)
    filled = int(width * ratio)
    return "█" * filled + "░" * (width - filled)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_progress_line(
    *,
    counts: BuildCounts,
    targets: BuildTargets,
    started_at: float,
    now: float | None = None,
) -> str:
    now = time.monotonic() if now is None else now
    elapsed = now - started_at
    overall = counts.total / targets.total if targets.total else 1.0
    eta = (elapsed / overall - elapsed) if 0.0 < overall < 1.0 else 0.0
    stamp = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    eta_text = _fmt_duration(eta) if eta > 0 else "done" if overall >= 1.0 else "?"
    return (
        f"[{stamp}] build {_bar(overall)} {overall * 100:5.1f}% | "
        f"hr {counts.hr_clip:4d}/{targets.hr_clip} | "
        f"codec {counts.codec:6d}/{targets.codec} | "
        f"elapsed {_fmt_duration(elapsed)} eta {eta_text}"
    )


def watch(
    root: Path,
    config_path: Path,
    *,
    interval: float,
    stream: object = sys.stderr,
) -> None:
    targets = load_targets(root, config_path)
    started_at = time.monotonic()
    stop = False

    def _handle_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    # Print initial progress at 0%
    print(
        format_progress_line(
            counts=BuildCounts(0, 0),
            targets=targets,
            started_at=started_at,
            now=started_at,
        ),
        file=stream,
        flush=True,
    )

    # Immediately check once before entering the sleep loop
    counts = count_outputs(root, config_path)
    if counts.total > 0:
        print(
            format_progress_line(counts=counts, targets=targets, started_at=started_at),
            file=stream,
            flush=True,
        )

    while not stop:
        time.sleep(interval)
        if stop:
            break
        counts = count_outputs(root, config_path)
        print(
            format_progress_line(counts=counts, targets=targets, started_at=started_at),
            file=stream,
            flush=True,
        )
        if counts.total >= targets.total:
            break

    # Final print: always show the actual completion state as a safeguard
    counts = count_outputs(root, config_path)
    print(
        format_progress_line(counts=counts, targets=targets, started_at=started_at),
        file=stream,
        flush=True,
    )
    if counts.total >= targets.total:
        print(f"[build_progress] All {targets.total} targets completed.", file=stream, flush=True)
    else:
        print(f"[build_progress] WARNING: completed {counts.total}/{targets.total} targets before exit.", file=stream, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Log tqdm-style build progress")
    parser.add_argument("--root", type=Path, default=_repo_root())
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "scripts/pipeline/config.yaml",
    )
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args(argv)
    watch(args.root.resolve(), args.config.resolve(), interval=args.interval)


if __name__ == "__main__":
    main()
