#!/usr/bin/env bash
# Shared setup for scripts/*.sh — source from other scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_PROXY="swanlab.cn,.swanlab.cn,api.swanlab.cn,localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

# SwanLab SDK parses SWANLAB_EXPERIMENT as JSON on import.
unset SWANLAB_EXPERIMENT SWANLAB_RESUME SWANLAB_RUN_ID

run_uv() {
  uv run "$@"
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
  if [[ -z "$resume" && -f "${save_dir}/best.pth" ]]; then
    resume="${save_dir}/best.pth"
  fi
  if [[ -n "$resume" && -f "$resume" ]]; then
    echo "[$(date -Is)] resuming from $resume" >&2
    RESUME_ARGS=(--resume "$resume")
  fi
}
