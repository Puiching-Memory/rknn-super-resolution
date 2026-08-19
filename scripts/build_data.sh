#!/usr/bin/env bash
# Build train/val manifests and offline codec cache via Snakemake.
#
# Usage:
#   ./scripts/build_data.sh
#   CLIPS_PER_VIDEO=4 BITRATES=150,200,300 ./scripts/build_data.sh
#   WORKERS=32 ./scripts/build_data.sh
#   ./scripts/build_data.sh -- -n   # dry-run (forwarded to snakemake)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WORKERS="${WORKERS:-$(nproc)}"
SNAKEFILE="${SNAKEFILE:-scripts/pipeline/Snakefile}"
CONFIG="${CONFIG:-scripts/pipeline/config.yaml}"
# Snakemake prints one block per job by default (~100k lines for full rebuild).
SNAKEMAKE_QUIET="${SNAKEMAKE_QUIET:-all}"

# Optional config overrides via environment.
SNAKEMAKE_CONFIG=()
if [[ -n "${CLIPS_PER_VIDEO:-}" ]]; then
  SNAKEMAKE_CONFIG+=(--config "clips_per_video=${CLIPS_PER_VIDEO}")
fi
if [[ -n "${BITRATES:-}" ]]; then
  SNAKEMAKE_CONFIG+=(--config "bitrates_kbps=[${BITRATES}]")
fi

echo "[$(date -Is)] discover sources (train + val manifests)"
run_uv python -m rk3588_mobile_sr.data_pipeline.discover_sources --root "$ROOT"

echo "[$(date -Is)] snakemake codec pipeline (workers=$WORKERS, quiet=$SNAKEMAKE_QUIET)"
SNAKEMAKE_ARGS=(-j "$WORKERS" -s "$SNAKEFILE" --directory "$ROOT" --configfile "$CONFIG" --rerun-incomplete)
if [[ -n "$SNAKEMAKE_QUIET" ]]; then
  SNAKEMAKE_ARGS+=(--quiet "$SNAKEMAKE_QUIET")
fi

PROGRESS_PID=""
if [[ "${BUILD_PROGRESS:-1}" != "0" ]]; then
  run_uv python -m rk3588_mobile_sr.data_pipeline.build_progress \
    --root "$ROOT" \
    --config "$CONFIG" \
    --interval "${BUILD_PROGRESS_INTERVAL:-60}" &
  PROGRESS_PID=$!
fi

_stop_progress() {
  if [[ -n "$PROGRESS_PID" ]]; then
    kill -TERM "$PROGRESS_PID" 2>/dev/null || true
    wait "$PROGRESS_PID" 2>/dev/null || true
    PROGRESS_PID=""
  fi
}
trap _stop_progress EXIT

set +e
run_uv snakemake \
  "${SNAKEMAKE_ARGS[@]}" \
  "${SNAKEMAKE_CONFIG[@]}" \
  -- \
  all \
  "$@"
SMK_EXIT=$?
set -e

_stop_progress
trap - EXIT

if [[ "$SMK_EXIT" -ne 0 ]]; then
  exit "$SMK_EXIT"
fi

echo "[$(date -Is)] done -> data/codec_cache/manifest.jsonl + data/raw_cache/*_{lr,hr}.npy"
