"""Snakemake pipeline integration tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.helpers.codec_fixture import (
    REPO_ROOT,
    build_snakemake_codec_fixture,
    run_snakemake_pipeline,
    write_test_config,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_snakemake_pipeline_smoke(tmp_path: Path):
    manifest = build_snakemake_codec_fixture(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["codec"] == "libx264"
    assert (tmp_path / rows[0]["path"]).is_file()


def test_snakemake_dry_run_cache_hit(tmp_path: Path):
    """Existing .npy outputs should make snakemake -n report nothing to do."""
    build_snakemake_codec_fixture(tmp_path)
    config_path = write_test_config(
        tmp_path,
        clips_per_video=1,
        clip_frames=4,
        lr_height=24,
        lr_width=32,
    )
    proc = __import__("subprocess").run(
        [
            "uv",
            "run",
            "snakemake",
            "-n",
            "-j",
            "1",
            "-s",
            str(REPO_ROOT / "scripts/pipeline/Snakefile"),
            "--directory",
            str(tmp_path),
            "--configfile",
            str(config_path),
            "--",
            "write_codec_manifest",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Nothing to be done" in proc.stdout + proc.stderr
