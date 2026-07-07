#!/usr/bin/env bash
# Build train/val manifests and offline codec cache.
#
# Usage:
#   ./scripts/build_data.sh
#   CLIPS_PER_VIDEO=4 BITRATES=150,200,300 ./scripts/build_data.sh
#   WORKERS=32 ./scripts/build_data.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CLIPS_PER_VIDEO="${CLIPS_PER_VIDEO:-8}"
CODECS="${CODECS:-libx264,libx265,libsvtav1}"
BITRATES="${BITRATES:-150,200,300,500,800}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/sources/manifests/train.jsonl}"
WORKERS="${WORKERS:-$(nproc)}"

echo "[$(date -Is)] building train manifest"
run_uv rk3588-build-train-manifest

echo "[$(date -Is)] building fixed val manifest"
run_uv rk3588-build-val-codec-fixed

echo "[$(date -Is)] building codec cache (clips=$CLIPS_PER_VIDEO codecs=$CODECS bitrates=$BITRATES workers=$WORKERS)"
run_uv rk3588-build-codec-cache \
  --sources "$TRAIN_MANIFEST" \
  --clips-per-video "$CLIPS_PER_VIDEO" \
  --codecs "$CODECS" \
  --bitrates "$BITRATES" \
  --workers "$WORKERS"

echo "[$(date -Is)] done -> data/codec_cache/manifest.jsonl"
