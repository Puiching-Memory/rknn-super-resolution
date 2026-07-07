#!/usr/bin/env bash
# Generate report charts from stage1_metrics.json -> docs/report_assets/.
#
# Usage:
#   ./scripts/generate_report_charts.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_uv rk3588-generate-report-charts "$@"
