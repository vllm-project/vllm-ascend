#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export LD_LIBRARY_PATH="${ASCEND_TOOLKIT_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/python/site-packages:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../../../vllm-ascend/"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../../../../vllm/"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

if [ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK="${VLLM_TORCH_PROFILER_WITH_STACK:-0}"
fi

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/../../../examples/fuxi_alpha/}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/hstu_prompts_d.jsonl}"
RESULT_DIR="${RESULT_DIR:-./log/}"
PORT="${PORT:-8100}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
REQUEST_RATE="${REQUEST_RATE:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

vllm bench serve \
     --model "${MODEL_PATH}" \
     --backend vllm \
     --port "${PORT}" \
     --save-result \
     --save-detailed \
     --endpoint /v1/completions \
     --dataset-name custom \
     --custom-skip-chat-template \
     --dataset-path "${DATASET_PATH}" \
     --num-prompts "${NUM_PROMPTS}" \
     --request_rate "${REQUEST_RATE}" \
     --burstiness 1 \
     --max-concurrency "${MAX_CONCURRENCY}" \
     --result-dir "${RESULT_DIR}" \
     --skip_tokenizer_init true
