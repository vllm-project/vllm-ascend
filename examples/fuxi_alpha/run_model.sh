#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${SCRIPT_DIR}/../../../mooncake.json}"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/../../../vllm-ascend/"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../../../vllm/"
export MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}}"
export DATASET_PATH="${DATASET_PATH:-/path/to/kuairand/dataset}"
if [ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]; then
    export VLLM_TORCH_PROFILER_WITH_STACK="${VLLM_TORCH_PROFILER_WITH_STACK:-0}"
fi
export KVCACHE_BACKEND="${KVCACHE_BACKEND:-mooncake}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

bash run.sh \
     --use_random 1 \
     --embedding_dim 4096 \
     --num_heads 16 \
     --dim 256 \
     --max_seq_len 2048 \
     --max_batch_size 2 \
     --aclgraph 1 \
     --candidate_num 256 \
     --has_ffn 1 \
     --max_vocab_size 4096 \
     --concat_batch 1 \
     --profiler 0 \
     --max_model_len 8832 \
     --range 2 \
     --graph_step 512 \
     --block_size 128
