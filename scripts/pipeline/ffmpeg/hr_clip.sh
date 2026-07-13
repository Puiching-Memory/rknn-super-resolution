#!/usr/bin/env bash
# Extract a clip from raw YUV420p as lossless H.264 (CRF 0) for clean HR
# supervision. Replaces the old CRF=18 mezzanine: the HR label is now bit-exact
# with the original YUV, so no compression artifacts leak into the target.
#
# Args: YUV_PATH OUTPUT WIDTH HEIGHT FPS CLIP_START CLIP_FRAMES
set -euo pipefail
YUV_PATH="$1"
OUTPUT="$2"
WIDTH="$3"
HEIGHT="$4"
FPS="$5"
CLIP_START="$6"
CLIP_FRAMES="$7"
END=$((CLIP_START + CLIP_FRAMES - 1))
mkdir -p "$(dirname "$OUTPUT")"
# rawvideo cannot -ss byte-seek reliably; select+setpts re-timelines the clip so
# the output frame index starts at 0 (matching the LR codec clip layout).
# -crf 0 = lossless; -pix_fmt yuv420p preserves the original chroma subsampling
# so NVDEC can decode it and the Y/U/V planes are bit-identical to the source.
ffmpeg -hide_banner -loglevel error -y \
  -f rawvideo -pix_fmt yuv420p -s "${WIDTH}x${HEIGHT}" -r "$FPS" -i "$YUV_PATH" \
  -vf "select='between(n\,${CLIP_START}\,${END})',setpts=PTS-STARTPTS" \
  -vsync 0 -frames:v "$CLIP_FRAMES" \
  -c:v libx264 -crf 0 -pix_fmt yuv420p -g "$CLIP_FRAMES" -preset veryfast \
  "$OUTPUT"
