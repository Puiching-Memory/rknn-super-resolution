"""CLI thin wrapper around the Snakemake data pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _snakefile() -> Path:
    return _repo_root() / "scripts" / "pipeline" / "Snakefile"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build offline codec LR clip cache via Snakemake"
    )
    parser.add_argument(
        "--sources",
        default="data/sources/manifests/train.jsonl",
        help="unused; train manifest path is in pipeline config.yaml",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--snake-args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to snakemake (e.g. -n for dry-run)",
    )
    args = parser.parse_args(argv)
    del args.sources

    root = _repo_root()
    snakefile = _snakefile()
    if not snakefile.is_file():
        raise SystemExit(f"Snakefile not found: {snakefile}")

    workers = args.workers if args.workers is not None else (os.cpu_count() or 4)
    cmd = [
        "uv",
        "run",
        "snakemake",
        "-j",
        str(workers),
        "-s",
        str(snakefile),
        "--directory",
        str(root),
        "--configfile",
        str(root / "scripts/pipeline/config.yaml"),
        "--rerun-incomplete",
        "--quiet",
        os.environ.get("SNAKEMAKE_QUIET", "all"),
    ]
    if args.snake_args:
        cmd.append("--")
        cmd.extend(args.snake_args)
    else:
        cmd.extend(["--", "all"])
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
