#!/usr/bin/env bash
# Download MambaIRv2Light ×3 teacher weights for Stage 2.
#
# Usage:
#   ./scripts/download_teacher.sh
#   https_proxy=http://127.0.0.1:7897 ./scripts/download_teacher.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

OUTPUT="${OUTPUT:-checkpoints/teacher/mambairv2_lightSR_x3.pth}"
PROXY="${https_proxy:-${HTTPS_PROXY:-}}"

ARGS=(--output "$OUTPUT")
if [[ -n "$PROXY" ]]; then
  ARGS+=(--proxy "$PROXY")
fi

run_uv rk3588-download-teacher "${ARGS[@]}"
