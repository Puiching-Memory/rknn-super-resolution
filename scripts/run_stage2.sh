#!/usr/bin/env bash
# Stage 2 distillation fine-tuning (DDP).
#
# Prerequisites:
#   ./scripts/download_teacher.sh
#   checkpoints/stage1/best.pth
#
# Environment overrides: SAVE_DIR, NPROC, TRAIN_EXPERIMENT_NAME, RESUME, EXTRA_ARGS
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NPROC="${NPROC:-8}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/stage2}"
STAGE1_WEIGHT="${STAGE1_WEIGHT:-./checkpoints/stage1/best.pth}"
TEACHER_WEIGHT="${TEACHER_WEIGHT:-./checkpoints/teacher/mambairv2_lightSR_x3.pth}"
CONSOLE_LOG="${SAVE_DIR}/console.log"
EXP_NAME="${TRAIN_EXPERIMENT_NAME:-stage2-${NPROC}gpu-$(date +%Y%m%d-%H%M)}"

for path in "$STAGE1_WEIGHT" "$TEACHER_WEIGHT"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required weight: $path" >&2
    exit 1
  fi
done

mkdir -p "$SAVE_DIR"
setup_resume_args "$SAVE_DIR"

echo "[$(date -Is)] starting stage2 ${NPROC}gpu experiment=$EXP_NAME" | tee -a "$CONSOLE_LOG"

exec >>"$CONSOLE_LOG" 2>&1
run_torchrun "$NPROC" -m rk3588_mobile_sr.train.stage2 \
  --codec_manifest data/codec_cache/manifest.jsonl \
  --val_manifest data/sources/manifests/val_fixed.jsonl \
  --decode auto \
  --batch_size 16 \
  --patch_size 160 \
  --max_steps 80000 \
  --val_every 4000 \
  --save_every 8000 \
  --log_every 500 \
  --lr 3e-5 \
  --teacher_arch mambairv2_light \
  --teacher_weight "$TEACHER_WEIGHT" \
  --stage1_weight "$STAGE1_WEIGHT" \
  --early_stop_patience 8 \
  --early_stop_min_delta 0.005 \
  --swanlab_project rk3588-mobile-sr \
  --swanlab_experiment "$EXP_NAME" \
  ${EXTRA_ARGS:-} \
  "${RESUME_ARGS[@]}"
