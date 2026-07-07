#!/usr/bin/env bash
# Benchmark codec canvas dataloader throughput (single GPU).
#
# Usage:
#   ./scripts/bench_dataloader.sh
#   BATCH_SIZE=8 DECODE=torchcodec ./scripts/bench_dataloader.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_uv rk3588-bench-dataloader \
  --codec_manifest "${CODEC_MANIFEST:-data/codec_cache/manifest.jsonl}" \
  --decode "${DECODE:-auto}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --steps "${STEPS:-100}" \
  --warmup "${WARMUP:-20}" \
  --device_id "${DEVICE_ID:-0}" \
  "$@"
