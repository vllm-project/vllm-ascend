#!/usr/bin/env bash
#
# GLM-5.1 decode-only sparse KV offload with TP and DP=1.
# The legacy filename is retained for compatibility. Speculative/MTP decoding
# is intentionally disabled until the DSA path passes device token-accuracy
# validation.
# Override these values for the target server topology before launching.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.1 model directory}"

DP_SIZE="${DP_SIZE:-1}"  # DSA supports DP>1; each DP rank owns independent DSA state
TP_SIZE="${TP_SIZE:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# Chunked prefill is intentionally disabled. This value must cover the
# longest prompt admitted by the service.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
# Quantized GLM-5/5.1 weights require the Ascend loader. Set QUANTIZATION to
# an empty string explicitly when serving a BF16 checkpoint.
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
# This migration/debug example emits the first-sample boundary by default when
# DSA is enabled. Set DSA_TRACE_ENABLED=false for performance measurements.
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
# Graph mode reuses the native FULL graph family for eligible single-token
# decode. When enabled, --enforce-eager is omitted so vLLM can capture/replay
# graphs. When disabled (default), --enforce-eager forces eager execution.
DSA_GRAPH_ENABLED="${DSA_GRAPH_ENABLED:-false}"
case "${DSA_GRAPH_ENABLED}" in
  true|false) ;;
  *)
    echo "DSA_GRAPH_ENABLED must be true or false, got: ${DSA_GRAPH_ENABLED}" >&2
    exit 2
    ;;
esac
DSA_ADDITIONAL_CONFIG="{\"dsa_sparse_config\":{\"enabled\":${DSA_ENABLED},\"indexer_mla_block_ratio\":3,\"enable_row_mode_decode_graph\":${DSA_GRAPH_ENABLED},\"trace_points\":{\"enabled\":${DSA_TRACE_ENABLED},\"points\":[\"first_sample\"],\"ranks\":[${DSA_TRACE_RANK}]}},\"enable_dsa_cp\":false,\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# When graph mode is disabled, force eager execution. When enabled, let
# vLLM choose the FULL graph path for eligible decode batches.
EXTRA_ARGS=()
if [[ "${DSA_GRAPH_ENABLED}" == "false" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --data-parallel-size "${DP_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
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
