"""RKNN conversion runs in a dedicated Python env (outside the project uv venv)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RKNN_PYTHON_ENV = "RKNN_PYTHON"
LEGACY_RKNN_PYTHON_ENVS = ("RK3576_RKNN_PYTHON", "RK3588_RKNN_PYTHON")
DEFAULT_RKNN_PYTHON = ".venv-rknn/bin/python"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_python_path(raw: str) -> Path:
    """Expand ``~``; paths relative to the repo root (directory with pyproject.toml)."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    # Keep the venv entrypoint path; do not resolve() symlinks to the uv base interpreter.
    return path.absolute()


def _default_rknn_python_raw() -> str:
    try:
        from rknn_super_resolution.config import load_config

        return load_config().deploy.rknn_python
    except ImportError:
        return DEFAULT_RKNN_PYTHON


def resolve_rknn_python(explicit: str | None = None) -> Path:
    """Resolve RKNN interpreter: CLI > env > YAML deploy.rknn_python > default."""
    if explicit:
        return _resolve_python_path(explicit)
    env = os.environ.get(RKNN_PYTHON_ENV)
    if env:
        return _resolve_python_path(env)
    for legacy_env in LEGACY_RKNN_PYTHON_ENVS:
        if env := os.environ.get(legacy_env):
            return _resolve_python_path(env)
    return _resolve_python_path(_default_rknn_python_raw())


def needs_rknn_reexec(python: Path) -> bool:
    expected_prefix = python.parent.parent.absolute()
    return Path(sys.prefix).absolute() != expected_prefix


def reexec_in_rknn_python(python: Path, argv: list[str]) -> None:
    """Run ``deploy.rknn`` under *python* with ``src/`` on PYTHONPATH."""
    if not python.is_file():
        raise SystemExit(
            f"RKNN Python not found: {python}\n"
            f"Create the env and install rknn-toolkit2, or set {RKNN_PYTHON_ENV} / deploy.rknn_python."
        )

    src = _src_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH")]))

    cmd = [str(python), "-m", "rknn_super_resolution.deploy.rknn", *argv]
    print(f"--> Re-exec with RKNN Python: {python}")
    raise SystemExit(subprocess.call(cmd, env=env, cwd=os.getcwd()))
