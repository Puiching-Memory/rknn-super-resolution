#!/usr/bin/env bash
# Multi-GPU DDP training step benchmark.
#
# Usage:
#   ./scripts/bench_ddp.sh
#   NPROC=4 BATCH_SIZE=8 ./scripts/bench_ddp.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NPROC="${NPROC:-8}"
run_torchrun "$NPROC" -m rk3588_mobile_sr.dev.bench_ddp \
  --codec_manifest "${CODEC_MANIFEST:-data/codec_cache/manifest.jsonl}" \
  --decode "${DECODE:-auto}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --steps "${STEPS:-50}" \
  --warmup "${WARMUP:-10}" \
  "$@"
