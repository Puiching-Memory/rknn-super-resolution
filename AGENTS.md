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

PyTorch and DALI wheel sources are pinned in `pyproject.toml` under `[tool.uv.index]` and `[tool.uv.sources]`.

### Notes

- Training, evaluation, and ONNX export require an NVIDIA GPU with a compatible driver.
- RKNN conversion (`rknn-toolkit2`) has its own torch/onnx version constraints. Keep it in a separate environment and install it manually when needed.
