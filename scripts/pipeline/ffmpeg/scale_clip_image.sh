#!/usr/bin/env bash
# Scale a single frame from image mezzanine MP4 to LR rgb24 raw.
set -euo pipefail
MEZZANINE="$1"
OUTPUT="$2"
FRAME_INDEX="$3"
LR_W="$4"
LR_H="$5"
FPS="$6"
mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel error -y \
  -i "$MEZZANINE" \
  -vf "select='eq(n\,${FRAME_INDEX})',scale=${LR_W}:${LR_H}:flags=area,format=rgb24" \
  -frames:v 1 -r "$FPS" \
  -f rawvideo -pix_fmt rgb24 \
  "$OUTPUT"
