"""Tests for RKNN dedicated Python resolution."""

from pathlib import Path

from rk3588_mobile_sr.deploy.rknn_env import (
    RKNN_PYTHON_ENV,
    _project_root,
    _resolve_python_path,
    resolve_rknn_python,
)


def test_resolve_rknn_python_explicit_absolute():
    path = resolve_rknn_python("/usr/bin/python3")
    assert path.is_absolute()
    assert "python" in path.name


def test_resolve_rknn_python_relative_to_project_root():
    rel = _resolve_python_path(".venv-rknn/bin/python")
    assert rel == (_project_root() / ".venv-rknn/bin/python").absolute()
    assert "venv-rknn" in str(rel)


def test_resolve_rknn_python_from_config(monkeypatch):
    monkeypatch.delenv(RKNN_PYTHON_ENV, raising=False)
    path = resolve_rknn_python(None)
    assert path == (_project_root() / ".venv-rknn/bin/python").absolute()


def test_needs_rknn_reexec_distinguishes_uv_venvs():
    from rk3588_mobile_sr.deploy.rknn_env import needs_rknn_reexec

    rknn_py = (_project_root() / ".venv-rknn/bin/python").absolute()
    # Main .venv and .venv-rknn may share the same resolved base interpreter.
    assert needs_rknn_reexec(rknn_py) is True


def test_resolve_rknn_python_env_overrides_config(monkeypatch):
    monkeypatch.setenv(RKNN_PYTHON_ENV, ".venv-rknn/bin/python")
    assert resolve_rknn_python(None) == resolve_rknn_python(".venv-rknn/bin/python")
