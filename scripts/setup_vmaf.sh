#!/usr/bin/env bash
# Build Netflix libvmaf (VMAF v1 models) into .local/ for validation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENDOR="$ROOT/third_party/vmaf"
PREFIX="$ROOT/.local"

if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/Netflix/vmaf.git "$VENDOR"
fi

uv pip install 'meson>=1.0' ninja
MESON="${MESON:-$ROOT/.venv/bin/meson}"
NINJA="${NINJA:-$ROOT/.venv/bin/ninja}"

cd "$VENDOR/libvmaf"
rm -rf build
"$MESON" setup build --buildtype release --prefix "$PREFIX" -Denable_float=true
"$NINJA" -C build -j"$(nproc)"
"$NINJA" -C build install

export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$PREFIX/bin/vmaf" --version
echo "Installed vmaf -> $PREFIX/bin/vmaf"
echo "VMAF v1 models (built-in version=...): default 1080p=vmaf_v1.0.16_3d0h phone=vmaf_v1.0.16_5d0h"
echo "Docs: https://github.com/Netflix/vmaf/blob/master/resource/doc/models_v1.md"
