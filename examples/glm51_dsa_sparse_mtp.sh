#!/usr/bin/env bash
#
# GLM-5.1 decode-only sparse KV offload, eager DP+TP, optional MTP.
# Override these values for the target server topology before launching.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.1 model directory}"

DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# Chunked prefill is intentionally disabled. This value must cover the
# longest prompt admitted by the service.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
PORT="${PORT:-8000}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --data-parallel-size "${DP_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --block-size 128 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --enforce-eager \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --speculative-config \
  "{\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"method\":\"deepseek_mtp\",\"enforce_eager\":true}" \
  --additional-config \
  '{"dsa_sparse_config":{"enabled":true,"split_indexer_cache":true,"indexer_mla_block_ratio":3},"ascend_compilation_config":{"enable_npugraph_ex":false}}'
