#!/usr/bin/env bash
# Profile stage-1 data loading vs compute (single GPU).
#
# Usage:
#   ./scripts/profile_stage1.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_uv rk3588-profile-stage1 \
  --codec_manifest "${CODEC_MANIFEST:-data/codec_cache/manifest.jsonl}" \
  --decode "${DECODE:-auto}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --device_id "${DEVICE_ID:-0}" \
  "$@"
