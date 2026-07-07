#!/usr/bin/env bash
# Scale a temporal clip from HR mezzanine MP4 to LR rgb24 raw.
set -euo pipefail
MEZZANINE="$1"
OUTPUT="$2"
CLIP_START="$3"
CLIP_FRAMES="$4"
LR_W="$5"
LR_H="$6"
FPS="$7"
END=$((CLIP_START + CLIP_FRAMES - 1))
mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel error -y \
  -i "$MEZZANINE" \
  -vf "select='between(n\,${CLIP_START}\,${END})',scale=${LR_W}:${LR_H}:flags=area,format=rgb24" \
  -vsync 0 \
  -f rawvideo -pix_fmt rgb24 \
  "$OUTPUT"
