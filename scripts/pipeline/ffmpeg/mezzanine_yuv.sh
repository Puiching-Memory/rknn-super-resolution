#!/usr/bin/env bash
# Encode raw YUV420p mezzanine from a full YUV source file.
set -euo pipefail
YUV_PATH="$1"
OUTPUT="$2"
WIDTH="$3"
HEIGHT="$4"
FPS="$5"
CRF="${6:-18}"
GOP="${7:-1}"
PRESET="${8:-veryfast}"
mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel error -y \
  -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$YUV_PATH" \
  -c:v libx264 -pix_fmt yuv420p -g "$GOP" -preset "$PRESET" -crf "$CRF" \
  "$OUTPUT"
