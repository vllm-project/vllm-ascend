#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/python/site-packages:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../../../vllm-ascend/"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../../../../vllm/"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-/path/to/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULT_DIR="${RESULT_DIR:-./log/}"
PORT="${PORT:-8105}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"

vllm bench serve \
     --model "${MODEL_PATH}" \
     --backend vllm \
     --port "${PORT}" \
     --save-result \
     --profile \
     --save-detailed \
     --endpoint /v1/completions \
     --dataset-name sharegpt \
     --dataset-path "${DATASET_PATH}" \
     --num-prompts "${NUM_PROMPTS}" \
     --result-dir "${RESULT_DIR}"
