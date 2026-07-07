"""Periodic tqdm-style progress lines for the Snakemake data build."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rk3588_mobile_sr.data_pipeline.clip_plan import (
    iter_encode_jobs,
    load_train_sources,
    unique_scale_jobs,
)


@dataclass(frozen=True)
class BuildTargets:
    mezzanine: int
    scaled: int
    codec: int

    @property
    def total(self) -> int:
        return self.mezzanine + self.scaled + self.codec


@dataclass(frozen=True)
class BuildCounts:
    mezzanine: int
    scaled: int
    codec: int

    @property
    def total(self) -> int:
        return self.mezzanine + self.scaled + self.codec


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
        gop=cfg["gop"],
        image_gop=cfg["image_gop"],
        seed=cfg["seed"],
    )
    return BuildTargets(
        mezzanine=len(sources),
        scaled=len(unique_scale_jobs(jobs)),
        codec=len(jobs),
    )


def _count_suffix_files(directory: Path, suffix: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix))


def count_outputs(root: Path, config_path: Path) -> BuildCounts:
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    mezzanine_dir = root / cfg["mezzanine_dir"]
    scaled_dir = root / cfg["scaled_dir"]
    codec_dir = root / cfg["codec_cache_dir"]
    return BuildCounts(
        mezzanine=_count_suffix_files(mezzanine_dir, "_hr.mp4"),
        scaled=_count_suffix_files(scaled_dir, ".rgb"),
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
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    eta_text = _fmt_duration(eta) if eta > 0 else "done" if overall >= 1.0 else "?"
    return (
        f"[{stamp}] build {_bar(overall)} {overall * 100:5.1f}% | "
        f"mezz {counts.mezzanine:4d}/{targets.mezzanine} | "
        f"scaled {counts.scaled:5d}/{targets.scaled} | "
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

    print(
        format_progress_line(
            counts=BuildCounts(0, 0, 0),
            targets=targets,
            started_at=started_at,
            now=started_at,
        ),
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

    counts = count_outputs(root, config_path)
    print(
        format_progress_line(counts=counts, targets=targets, started_at=started_at),
        file=stream,
        flush=True,
    )


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
