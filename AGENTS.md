# Project Guidelines

## Python Environment

This project uses `uv` as the default Python package and environment manager. Do not use `pip` directly unless the user explicitly asks for it.

- Python version: `3.12.*` (defined in `pyproject.toml`).
- GPU stack: CUDA 13.0 (`cu130`) PyTorch wheels only; CPU-only PyTorch is not supported.
- Lock file: `uv.lock` is committed; prefer `uv sync` to keep it consistent.

### Install

```bash
uv sync
```

For development (lint, tests, pre-commit):

```bash
uv sync
```

PyTorch and DALI wheel sources are pinned in `pyproject.toml` under `[tool.uv.index]` and `[tool.uv.sources]`.

### Project Layout

The installable package lives under `src/rk3588_mobile_sr/` (src layout). Entry points:

- `rk3588-mobile-sr` — unified CLI (`train`, `eval`, `export-onnx`, `convert-rknn`)
- Legacy aliases: `rk3588-train-stage1`, `rk3588-train-stage2`, etc.

### Notes

- Training, evaluation, and ONNX export require an NVIDIA GPU with a compatible driver.
- RKNN conversion (`rknn-toolkit2`) has its own torch/onnx version constraints. Keep it in a separate environment and install it manually when needed.
