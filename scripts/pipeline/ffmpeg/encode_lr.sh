#!/usr/bin/env bash
# Encode rgb24 raw clip to LR codec MP4.
set -euo pipefail
RAW="$1"
OUTPUT="$2"
WIDTH="$3"
HEIGHT="$4"
FPS="$5"
FRAMES="$6"
CODEC="$7"
GOP="$8"
BITRATE="$9"
PRESET="${10:-veryfast}"
mkdir -p "$(dirname "$OUTPUT")"
uv run python - "$RAW" "$OUTPUT" "$WIDTH" "$HEIGHT" "$FPS" "$FRAMES" "$CODEC" "$GOP" "$BITRATE" "$PRESET" <<'PY'
import sys
from pathlib import Path

from rk3588_mobile_sr.data_pipeline.codec_args import encode_rgb_raw_to_mp4

encode_rgb_raw_to_mp4(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    width=int(sys.argv[3]),
    height=int(sys.argv[4]),
    fps=int(sys.argv[5]),
    frames=int(sys.argv[6]),
    codec=sys.argv[7],
    gop=int(sys.argv[8]),
    bitrate_kbps=int(sys.argv[9]),
    preset=sys.argv[10],
)
PY
