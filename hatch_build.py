"""Compile Netflix libvmaf into ``.local/`` during editable ``uv sync``."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Workspace hook: build the ``vmaf`` CLI next to the repo, not into the wheel."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        del build_data
        if self.target_name == "sdist":
            return
        if version not in {"editable", "standard"}:
            return

        root = Path(self.root)
        script = root / "scripts" / "setup_vmaf.sh"
        meson_build = root / "third_party" / "vmaf" / "libvmaf" / "meson.build"
        if not script.is_file():
            raise RuntimeError(f"missing {script}")
        if not meson_build.is_file():
            raise RuntimeError(
                "Netflix libvmaf source is required "
                "(git submodule third_party/vmaf). "
                "Clone with --recurse-submodules or run: "
                "git submodule update --init --recursive -- third_party/vmaf"
            )

        proc = subprocess.run(
            ["bash", str(script)],
            cwd=root,
            env=os.environ.copy(),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "libvmaf build failed; need nasm, xxd, meson, and ninja. "
                "meson/ninja come from [build-system] requires on `uv sync`. "
                f"See {script}"
            )
