#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ROOT_PATH="${ROOT_PATH:-./dataset/traffic/}"
DATA_PATH="${DATA_PATH:-traffic.csv}"
USE_AMP="${USE_AMP:-1}"

AMP_ARGS=()
if [[ "$USE_AMP" == "1" ]]; then
    AMP_ARGS+=(--use_amp)
fi

if [[ "$#" -gt 0 ]]; then
    HORIZONS=("$@")
else
    HORIZONS=(96 192 336 720)
fi

for H in "${HORIZONS[@]}"; do
    case "$H" in
        96)
            D_MODEL=64
            PATH_RANK=64
            DROPOUT=0.3
            LEARNING_RATE=5e-3
            BATCH_SIZE=64
            ;;
        192)
            D_MODEL=96
            PATH_RANK=64
            DROPOUT=0.2
            LEARNING_RATE=8e-3
            BATCH_SIZE=32
            ;;
        336)
            D_MODEL=32
            PATH_RANK=64
            DROPOUT=0.2
            LEARNING_RATE=8e-3
            BATCH_SIZE=32
            ;;
        720)
            D_MODEL=32
            PATH_RANK=64
            DROPOUT=0.2
            LEARNING_RATE=8e-3
            BATCH_SIZE=32
            ;;
        *)
            echo "Unsupported prediction horizon: $H"
            echo "Supported horizons: 96, 192, 336, 720"
            exit 1
            ;;
    esac

    echo "============================================================"
    echo "Traffic | H=$H | d_model=$D_MODEL | rank=$PATH_RANK | dropout=$DROPOUT"
    echo "============================================================"

    python -u run_tpfuse.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "$ROOT_PATH" \
      --data_path "$DATA_PATH" \
      --model_id "traffic_${H}" \
      --model TPFuse \
      --data custom \
      --features M \
      --target OT \
      --freq h \
      --seq_len 96 \
      --pred_len "$H" \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --d_model "$D_MODEL" \
      --path_rank "$PATH_RANK" \
      --dropout "$DROPOUT" \
      --learning_rate "$LEARNING_RATE" \
      --train_epochs 20 \
      --patience 5 \
      --batch_size "$BATCH_SIZE" \
      --lradj type3 \
      --itr 1 \
      "${AMP_ARGS[@]}"

done
