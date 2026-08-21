#!/usr/bin/env bash
# Init the MLVC submodule, then fetch MLVC-S weights and the OpenVidHD index.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MLVC_ROOT="${MLVC_ROOT:-third_party/mlvc}"
CHECKPOINT="${MLVC_CHECKPOINT:-data/mlvc/mlvc-s-psnr-v1.ckpt}"
CHECKPOINT_URL="https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-s-psnr-v1.ckpt"
CHECKPOINT_SHA256="1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896"
SEQUENCE_CSV="${OPENVID_SEQUENCE_CSV:-data/OpenVidHD/openvidhd_60k64_frame_sequences.csv}"
SEQUENCE_URL="https://mlvideopub.blob.core.windows.net/mlvc/datasets/OpenVidHD_parts_10-28/openvidhd_60k64_frame_sequences.csv"

if [[ "$MLVC_ROOT" == "third_party/mlvc" ]]; then
  ensure_submodule "third_party/mlvc"
fi
if [[ ! -d "${MLVC_ROOT}/video/src" ]]; then
  echo "error: MLVC source not found at ${MLVC_ROOT}/video/src" >&2
  exit 1
fi

mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$SEQUENCE_CSV")"
if [[ ! -f "$CHECKPOINT" ]]; then
  curl --fail --location --retry 3 "$CHECKPOINT_URL" --output "$CHECKPOINT"
fi
echo "$CHECKPOINT_SHA256  $CHECKPOINT" | sha256sum --check --status

if [[ ! -f "$SEQUENCE_CSV" ]]; then
  curl --fail --location --retry 3 "$SEQUENCE_URL" --output "$SEQUENCE_CSV"
fi

cat <<EOF
MLVC ready: $MLVC_ROOT @ $(git -C "$MLVC_ROOT" rev-parse --short HEAD)
checkpoint: $CHECKPOINT
sequence index: $SEQUENCE_CSV

OpenVidHD frames are intentionally not downloaded by this script. Download parts 10-28
from https://huggingface.co/datasets/nkp37/OpenVid-1M/tree/main/OpenVidHD, then run
MLVC's video/extract_frame_sequences.py and video/build_dataset_description.py to create:
  data/OpenVidHD/openvidhd_60k_train64/description.json
EOF
