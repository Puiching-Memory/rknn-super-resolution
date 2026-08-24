#!/usr/bin/env bash
# Unified plateau-driven FP32 -> QAT training. GPU count comes from flags / visible devices.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
  cat <<'EOF'
Usage: run_train.sh [script options] [--] [training args...]

Script options:
  -n, --nproc N          torchrun processes (default: number of visible GPUs)
  -d, --devices LIST     CUDA_VISIBLE_DEVICES, e.g. 0,1,3,4
  -s, --save-dir DIR     Checkpoint root (default: ./checkpoints/train)
  -e, --experiment NAME  SwanLab experiment name
  -h, --help             Show this help

Any other arguments are forwarded to rk3588_mobile_sr.train.unified.
Environment: NPROC, CUDA_VISIBLE_DEVICES, SAVE_DIR, RESUME,
TRAIN_EXPERIMENT_NAME, EXTRA_ARGS.

Examples:
  ./scripts/run_train.sh
  ./scripts/run_train.sh --devices 0,1,3,4,5,6
  ./scripts/run_train.sh --nproc 2 --save-dir checkpoints/smoke
  RESUME=checkpoints/train/last.pth ./scripts/run_train.sh --nproc 6
EOF
}

csv_count() {
  local spec="${1:-}"
  if [[ -z "$spec" ]]; then
    echo 0
    return
  fi
  local IFS=','
  local -a parts
  read -r -a parts <<< "$spec"
  echo "${#parts[@]}"
}

visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    csv_count "$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
    return
  fi
  echo 0
}

NPROC="${NPROC:-}"
DEVICES=""
SAVE_DIR="${SAVE_DIR:-./checkpoints/train}"
EXP_NAME="${TRAIN_EXPERIMENT_NAME:-}"
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -n | --nproc)
      NPROC="${2:?--nproc requires a value}"
      shift 2
      ;;
    --nproc=*)
      NPROC="${1#*=}"
      shift
      ;;
    -d | --devices)
      DEVICES="${2:?--devices requires a value}"
      shift 2
      ;;
    --devices=*)
      DEVICES="${1#*=}"
      shift
      ;;
    -s | --save-dir | --save_dir)
      SAVE_DIR="${2:?--save-dir requires a value}"
      shift 2
      ;;
    --save-dir=* | --save_dir=*)
      SAVE_DIR="${1#*=}"
      shift
      ;;
    -e | --experiment)
      EXP_NAME="${2:?--experiment requires a value}"
      shift 2
      ;;
    --experiment=*)
      EXP_NAME="${1#*=}"
      shift
      ;;
    --)
      shift
      FORWARD+=("$@")
      break
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_arr=($EXTRA_ARGS)
  FORWARD+=("${extra_arr[@]}")
fi

if [[ -n "$DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$DEVICES"
fi

if [[ -z "$NPROC" ]]; then
  NPROC="$(visible_gpu_count)"
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: nproc must be a positive integer, got ${NPROC:-empty}" >&2
  echo "pass --nproc N or --devices 0,1,... (and ensure nvidia-smi sees GPUs)" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible="$(csv_count "$CUDA_VISIBLE_DEVICES")"
  if (( NPROC > visible )); then
    echo "error: --nproc $NPROC exceeds visible GPUs (${CUDA_VISIBLE_DEVICES})" >&2
    exit 1
  fi
fi

CONSOLE_LOG="${SAVE_DIR}/console.log"
if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="train-${NPROC}gpu-$(date +%Y%m%d-%H%M)"
fi

mkdir -p "$SAVE_DIR"
setup_resume_args "$SAVE_DIR"

uses_vmaf=1
i=0
while (( i < ${#FORWARD[@]} )); do
  case "${FORWARD[$i]}" in
    --val_metric)
      next="${FORWARD[$((i + 1))]:-}"
      if [[ "${next,,}" == "psnr" ]]; then
        uses_vmaf=0
      fi
      ;;
    --val_metric=psnr)
      uses_vmaf=0
      ;;
  esac
  i=$((i + 1))
done
if (( uses_vmaf )); then
  ensure_vmaf
fi

# Temporary host workaround: translated IOMMU blocks PCIe peer DMA on this machine.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

echo "[$(date -Is)] starting unified training ${NPROC}gpu experiment=$EXP_NAME devices=${CUDA_VISIBLE_DEVICES:-all} nccl_p2p_disable=$NCCL_P2P_DISABLE" | tee -a "$CONSOLE_LOG"

exec >>"$CONSOLE_LOG" 2>&1
run_torchrun "$NPROC" -m rk3588_mobile_sr.train.unified \
  --dataset_description data/OpenVidHD/openvidhd_60k64_frame_sequences.csv \
  --video_root data/OpenVidHD \
  --mlvc_repo third_party/mlvc \
  --mlvc_checkpoint data/mlvc/mlvc-s-psnr-v1.ckpt \
  --val_every 1000 \
  --save_every 5000 \
  --log_every 500 \
  --swanlab_project rk3588-mobile-sr \
  --swanlab_experiment "$EXP_NAME" \
  --save_dir "$SAVE_DIR" \
  "${FORWARD[@]}" \
  "${RESUME_ARGS[@]}"
