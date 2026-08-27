#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RKNN_PYTHON="${PROJECT_ROOT}/.venv-rknn/bin/python"

cd "${PROJECT_ROOT}"
if [[ ! -x "${RKNN_PYTHON}" ]]; then
  uv venv --python 3.12 .venv-rknn
fi

uv pip install \
  --python "${RKNN_PYTHON}" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.4.0'
uv pip install \
  --python "${RKNN_PYTHON}" \
  'onnx==1.18.0' \
  'pyyaml==6.0.3' \
  'rknn-toolkit2==2.3.2'
uv pip check --python "${RKNN_PYTHON}"

"${RKNN_PYTHON}" - <<'PY'
import importlib.metadata

from rknn.api import RKNN

print(f"RKNN Toolkit2 {importlib.metadata.version('rknn-toolkit2')} ready: {RKNN}")
PY
