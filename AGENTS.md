# Project Guidelines

## Python Environment

This project uses `uv` as the default Python package and environment manager. Do not use `pip` directly unless the user explicitly asks for it.

- Python version: `3.12.*` (defined in `pyproject.toml`).
- Lock file: `uv.lock` is committed; prefer `uv sync` to keep it consistent.

### Install

```bash
# CPU variant (default)
uv sync --extra cpu

# CUDA variant
uv sync --extra cuda --extra-index-url https://download.pytorch.org/whl/cu126
```

### Notes

- PyTorch CPU and CUDA wheel sources are mutually exclusive; the CPU source is pinned in `pyproject.toml` under `[tool.uv]`, while CUDA must be specified on the command line.
- RKNN conversion (`rknn-toolkit2`) has its own torch/onnx version constraints. Keep it in a separate environment and install it manually when needed.
