#!/usr/bin/env bash
# Shared setup for scripts/*.sh — source from other scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_PROXY="swanlab.cn,.swanlab.cn,api.swanlab.cn,localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

# Project-local Netflix libvmaf (VMAF v1); built by hatch_build.py on `uv sync`.
export PATH="$ROOT/.local/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ROOT/.local/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# SwanLab SDK parses SWANLAB_EXPERIMENT as JSON on import.
unset SWANLAB_EXPERIMENT SWANLAB_RESUME SWANLAB_RUN_ID

ensure_submodule() {
  local path="${1:?submodule path required}"
  if [[ ! -f "$ROOT/.gitmodules" ]]; then
    echo "error: missing .gitmodules; clone with --recurse-submodules" >&2
    exit 1
  fi
  git -C "$ROOT" submodule update --init --recursive -- "$path"
}

run_uv() {
  uv run "$@"
}

ensure_vmaf() {
  local bin="$ROOT/.local/bin/vmaf"
  if [[ -x "$bin" ]]; then
    return 0
  fi
  echo "libvmaf CLI missing at $bin; building via scripts/setup_vmaf.sh" >&2
  if command -v meson >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
    "$ROOT/scripts/setup_vmaf.sh"
    return
  fi
  uv run --with 'meson>=1.0' --with ninja -- bash "$ROOT/scripts/setup_vmaf.sh"
}

run_torchrun() {
  local nproc="${1:?nproc required}"
  shift
  run_uv torchrun --nproc_per_node="$nproc" "$@"
}

RESUME_ARGS=()

setup_resume_args() {
  local save_dir="${1:?save_dir required}"
  RESUME_ARGS=()
  local resume="${RESUME:-}"
  if [[ -z "$resume" ]]; then
    local candidate
    for candidate in \
      "${save_dir}/last.pth" \
      "${save_dir}/qat_observe/last.pth" \
      "${save_dir}/float/last.pth"; do
      if [[ -f "$candidate" ]]; then
        resume="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$resume" && -f "$resume" ]]; then
    echo "[$(date -Is)] resuming from $resume" >&2
    RESUME_ARGS=(--resume "$resume")
  fi
}
