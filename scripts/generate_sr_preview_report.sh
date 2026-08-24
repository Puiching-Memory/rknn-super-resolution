#!/usr/bin/env bash
# Regenerate sr_preview panels + labeled grid for a run's report_charts/.
#
# Usage:
#   ./scripts/generate_sr_preview_report.sh --run_dir checkpoints/phase-rlfn-codec-v1
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_uv rknn-super-resolution-sr-preview-report "$@"
