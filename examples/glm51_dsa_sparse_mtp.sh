#!/usr/bin/env bash
#
# GLM-5.1 DSA sparse KV offload with TP+DP and optional MTP.
# Override these values for the target server topology before launching.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.1 model directory}"

DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-204800}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
QUANTIZATION="${QUANTIZATION-ascend}"
QUANTIZATION_ARGS=()
if [[ -n "${QUANTIZATION}" ]]; then
  QUANTIZATION_ARGS=(--quantization "${QUANTIZATION}")
fi
PORT="${PORT:-8000}"

DSA_ENABLED="${DSA_ENABLED:-true}"
case "${DSA_ENABLED}" in
  true|false) ;;
  *)
    echo "DSA_ENABLED must be true or false, got: ${DSA_ENABLED}" >&2
    exit 2
    ;;
esac

DSA_TRACE_ENABLED="${DSA_TRACE_ENABLED:-${DSA_ENABLED}}"
case "${DSA_TRACE_ENABLED}" in
  true|false) ;;
  *)
    echo "DSA_TRACE_ENABLED must be true or false, got: ${DSA_TRACE_ENABLED}" >&2
    exit 2
    ;;
esac

DSA_TRACE_RANK="${DSA_TRACE_RANK:-0}"
if [[ ! "${DSA_TRACE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "DSA_TRACE_RANK must be a non-negative TP rank, got: ${DSA_TRACE_RANK}" >&2
  exit 2
fi

DSA_GRAPH_ENABLED="${DSA_GRAPH_ENABLED:-false}"
case "${DSA_GRAPH_ENABLED}" in
  true|false) ;;
  *)
    echo "DSA_GRAPH_ENABLED must be true or false, got: ${DSA_GRAPH_ENABLED}" >&2
    exit 2
    ;;
esac

MTP_ENABLED="${MTP_ENABLED:-false}"
case "${MTP_ENABLED}" in
  true|false) ;;
  *)
    echo "MTP_ENABLED must be true or false, got: ${MTP_ENABLED}" >&2
    exit 2
    ;;
esac

DSA_ADDITIONAL_CONFIG="{\"dsa_sparse_config\":{\"enabled\":${DSA_ENABLED},\"indexer_mla_block_ratio\":3,\"enable_row_mode_decode_graph\":${DSA_GRAPH_ENABLED},\"trace_points\":{\"enabled\":${DSA_TRACE_ENABLED},\"points\":[\"first_sample\"],\"ranks\":[${DSA_TRACE_RANK}]}},\"enable_dsa_cp\":false,\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ "${DSA_GRAPH_ENABLED}" == "false" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

if [[ "${MTP_ENABLED}" == "true" ]]; then
  EXTRA_ARGS+=(--speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp"}')
fi

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --data-parallel-size "${DP_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --enable-expert-parallel \
  "${QUANTIZATION_ARGS[@]}" \
  --block-size 128 \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --additional-config \
  "${DSA_ADDITIONAL_CONFIG}"
