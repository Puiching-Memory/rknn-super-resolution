#!/usr/bin/env bash
# Unified plateau-driven FP32 -> QAT training on 8 GPUs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NPROC="${NPROC:-8}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/train}"
CONSOLE_LOG="${SAVE_DIR}/console.log"
EXP_NAME="${TRAIN_EXPERIMENT_NAME:-train-${NPROC}gpu-$(date +%Y%m%d-%H%M)}"

mkdir -p "$SAVE_DIR"
setup_resume_args "$SAVE_DIR"

echo "[$(date -Is)] starting unified training ${NPROC}gpu experiment=$EXP_NAME" | tee -a "$CONSOLE_LOG"

exec >>"$CONSOLE_LOG" 2>&1
run_torchrun "$NPROC" -m rk3588_mobile_sr.train.unified \
  --dataset_description data/OpenVidHD/openvidhd_60k_train64/description.json \
  --mlvc_repo third_party/mlvc \
  --mlvc_checkpoint data/mlvc/mlvc-s-psnr-v1.ckpt \
  --val_every 1000 \
  --save_every 5000 \
  --log_every 500 \
  --swanlab_project rk3588-mobile-sr \
  --swanlab_experiment "$EXP_NAME" \
  --save_dir "$SAVE_DIR" \
  ${EXTRA_ARGS:-} \
  "${RESUME_ARGS[@]}"
