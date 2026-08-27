"""Tests for RKNN export helpers (encryption paths)."""

from pathlib import Path

from rknn_super_resolution.deploy.rknn import _default_encrypted_output, _resolve_encrypted_output


def test_default_encrypted_output():
    assert _default_encrypted_output(Path("out/phase_rlfn_sr_x3.rknn")) == Path(
        "out/phase_rlfn_sr_x3.crypt.rknn"
    )


def test_resolve_encrypted_output_explicit_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = Path("/tmp/phase_rlfn_sr_x3.rknn")
    enc = _resolve_encrypted_output(out, "secure/model.enc.rknn")
    assert enc == (tmp_path / "secure/model.enc.rknn").resolve()
