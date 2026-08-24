#!/usr/bin/env bash
# Build Netflix libvmaf (VMAF v1) into .local/. Called by hatch_build.py on `uv sync`.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

VENDOR="$ROOT/third_party/vmaf"
PREFIX="$ROOT/.local"
STAMP="$PREFIX/vmaf-revision"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "error: missing '$name' (required to build libvmaf)" >&2
    echo "  Ubuntu/Debian: apt install nasm xxd ninja-build" >&2
    echo "  meson is a hatch build-system requirement; it is on PATH during uv sync." >&2
    exit 1
  fi
}

require_cmd nasm
require_cmd xxd

MESON="${MESON:-$(command -v meson || true)}"
NINJA="${NINJA:-$(command -v ninja || true)}"
if [[ -z "$MESON" ]]; then
  echo "error: meson not found on PATH" >&2
  echo "  uv sync builds libvmaf via hatch_build.py (meson is in [build-system] requires)." >&2
  echo "  Or: uv run --with 'meson>=1.0' --with ninja -- $0" >&2
  exit 1
fi
if [[ -z "$NINJA" ]]; then
  echo "error: ninja not found on PATH" >&2
  exit 1
fi

ensure_submodule "third_party/vmaf"

if [[ ! -f "$VENDOR/libvmaf/meson.build" ]]; then
  echo "error: $VENDOR/libvmaf/meson.build missing; initialize the vmaf submodule" >&2
  exit 1
fi

REV="unknown"
if git -C "$VENDOR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REV="$(git -C "$VENDOR" rev-parse HEAD)"
fi

if [[ "${FORCE_VMAF_REBUILD:-0}" != "1" && -x "$PREFIX/bin/vmaf" ]]; then
  if [[ -f "$STAMP" && "$(cat "$STAMP")" != "$REV" && "$REV" != "unknown" ]]; then
    echo "libvmaf source changed ($(cat "$STAMP") -> $REV); rebuilding"
  else
    echo "vmaf already installed -> $PREFIX/bin/vmaf ($("$PREFIX/bin/vmaf" --version 2>/dev/null || echo ok))"
    exit 0
  fi
fi

mkdir -p "$PREFIX"

cd "$VENDOR/libvmaf"
rm -rf build
"$MESON" setup build --buildtype release --prefix "$PREFIX" -Denable_float=true
"$NINJA" -C build -j"$(nproc)"
"$NINJA" -C build install

export PATH="$PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$PREFIX/bin/vmaf" --version
printf '%s\n' "$REV" >"$STAMP"
echo "Installed vmaf -> $PREFIX/bin/vmaf"
echo "VMAF v1 models (built-in version=...): default 1080p=vmaf_v1.0.16_3d0h phone=vmaf_v1.0.16_5d0h"
echo "Docs: https://github.com/Netflix/vmaf/blob/master/resource/doc/models_v1.md"
