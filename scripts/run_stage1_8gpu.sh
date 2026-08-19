#!/usr/bin/env bash
# Stage 1 DDP training on 8 GPUs.
#
# Environment overrides:
#   SAVE_DIR              checkpoint directory (default: ./checkpoints/stage1)
#   TRAIN_EXPERIMENT_NAME SwanLab experiment name
#   RESUME                checkpoint path to resume from
#   EXTRA_ARGS            extra args passed to training (quoted string)
#
# Usage:
#   ./scripts/run_stage1_8gpu.sh
#   RESUME=./checkpoints/stage1/best.pth ./scripts/run_stage1_8gpu.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NPROC="${NPROC:-8}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/stage1}"
CONSOLE_LOG="${SAVE_DIR}/console.log"
EXP_NAME="${TRAIN_EXPERIMENT_NAME:-stage1-${NPROC}gpu-$(date +%Y%m%d-%H%M)}"

mkdir -p "$SAVE_DIR"
setup_resume_args "$SAVE_DIR"

echo "[$(date -Is)] starting stage1 ${NPROC}gpu experiment=$EXP_NAME" | tee -a "$CONSOLE_LOG"

exec >>"$CONSOLE_LOG" 2>&1
run_torchrun "$NPROC" -m rk3588_mobile_sr.train.stage1 \
  --dataset_description data/OpenVidHD/openvidhd_60k_train64/description.json \
  --mlvc_repo third_party/mlvc \
  --mlvc_checkpoint data/mlvc/mlvc-s-psnr-v1.ckpt \
  --batch_size 2 \
  --patch_size 128 \
  --max_steps 100000 \
  --val_every 1000 \
  --save_every 5000 \
  --log_every 500 \
  --lr 1e-3 \
  --early_stop_patience 10 \
  --early_stop_min_delta 0.1 \
  --vmaf_model 1080p \
  --swanlab_project rk3588-mobile-sr \
  --swanlab_experiment "$EXP_NAME" \
  ${EXTRA_ARGS:-} \
  "${RESUME_ARGS[@]}"
