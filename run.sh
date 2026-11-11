#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GPU_ID=${1:-0}
MODEL=${2:-qwen3_vl}
DATA_PATH=${3:?Provide full RAH-Bench response pairs as argument 3}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u prepare.py \
    --model "$MODEL" --data_path "$DATA_PATH" \
    --max_samples 3000 --num_components 10 --device cuda

python -u train.py \
    --model "$MODEL" --num_components 10 \
    --epochs 200 --lr 1e-3 --device cuda

for DATASET in POPE MME AMBER CHAIR; do
    python -u evaluate.py \
        --model "$MODEL" --dataset "$DATASET" \
        --alpha 2.0 --routing_weight 0.5 --num_components 10 \
        --device cuda
done
