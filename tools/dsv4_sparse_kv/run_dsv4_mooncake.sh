#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/tmp/dsv4_sparse_kv/${timestamp}}"
mkdir -p "${OUT_DIR}"

MODEL_PATH="${MODEL_PATH:-/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"
PORT="${PORT:-8900}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50898}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-9090}"
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-15GB}"
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-${OUT_DIR}/mooncake.json}"
START_MOONCAKE="${START_MOONCAKE:-1}"
ENABLE_KV_POOL="${ENABLE_KV_POOL:-1}"
DSV4_EXPERIMENT_MODE="${DSV4_EXPERIMENT_MODE:-baseline}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v4}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_v4}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
MODEL_LOADER_EXTRA_CONFIG="${MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":\"true\",\"num_threads\":128}}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"num_speculative_tokens\":1,\"method\":\"mtp\",\"enforce_eager\":true}}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":false},\"enable_cpu_binding\":true,\"multistream_overlap_shared_expert\":true}}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,12,13}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export MOONCAKE_CONFIG_PATH
mkdir -p "$(dirname "${MOONCAKE_CONFIG_PATH}")"

JEMALLOC_PATH="${JEMALLOC_PATH:-/usr/lib64/libjemalloc.so.2}"
if [[ -e "${JEMALLOC_PATH}" ]]; then
    if [[ -n "${LD_PRELOAD:-}" ]]; then
        export LD_PRELOAD="${JEMALLOC_PATH}:${LD_PRELOAD}"
    else
        export LD_PRELOAD="${JEMALLOC_PATH}"
    fi
fi

if [[ "${WRITE_MOONCAKE_CONFIG:-1}" == "1" ]]; then
    cat > "${MOONCAKE_CONFIG_PATH}" <<JSON
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
    "global_segment_size": "${MOONCAKE_GLOBAL_SEGMENT_SIZE}",
    "use_ascend_direct": true
}
JSON
fi

MOONCAKE_PID=""
if [[ "${START_MOONCAKE}" == "1" ]]; then
    mooncake_master \
        --port "${MOONCAKE_MASTER_PORT}" \
        --metrics_port "${MOONCAKE_METRICS_PORT}" \
        --eviction_high_watermark_ratio "${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO:-0.9}" \
        --eviction_ratio "${MOONCAKE_EVICTION_RATIO:-0.1}" \
        --default_kv_lease_ttl "${MOONCAKE_DEFAULT_KV_LEASE_TTL:-11000}" \
        2>&1 | tee "${OUT_DIR}/run_mooncake.log" &
    MOONCAKE_PID="$!"
    trap 'if [[ -n "${MOONCAKE_PID}" ]]; then kill "${MOONCAKE_PID}" 2>/dev/null || true; fi' EXIT
    sleep "${MOONCAKE_STARTUP_WAIT:-3}"
fi

graph_args=()
case "${DSV4_EXPERIMENT_MODE}" in
    baseline)
        graph_args+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
        ;;
    eager_baseline)
        graph_args+=(--enforce-eager)
        ;;
    topk_trace)
        export VLLM_ASCEND_DSV4_TOPK_TRACE_ENABLE=1
        export VLLM_ASCEND_DSV4_TOPK_TRACE_PATH="${TOPK_TRACE_PATH:-${OUT_DIR}/topk_trace.jsonl}"
        export VLLM_ASCEND_DSV4_TOPK_TRACE_MAX_ROWS="${TOPK_TRACE_MAX_ROWS:-50000}"
        export VLLM_ASCEND_DSV4_TOPK_TRACE_SAMPLE_ROWS="${TOPK_TRACE_SAMPLE_ROWS:-1}"
        if [[ "${USE_ACLGRAPH:-0}" == "1" ]]; then
            graph_args+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
        else
            graph_args+=(--enforce-eager)
        fi
        ;;
    *)
        echo "Unknown DSV4_EXPERIMENT_MODE=${DSV4_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac

kv_args=()
if [[ "${ENABLE_KV_POOL}" == "1" ]]; then
    kv_args+=(--enable-prefix-caching)
    kv_args+=(--no-disable-hybrid-kv-cache-manager)
    kv_args+=(
        --kv-transfer-config
        '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"backend":"mooncake","lookup_rpc_port":"0","load_async":true}}'
    )
fi

serve_args=(
    serve "${MODEL_PATH}"
    "${kv_args[@]}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --api-server-count "${API_SERVER_COUNT}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --data-parallel-size "${DP_SIZE}"
    --tensor-parallel-size "${TP_SIZE}"
    --tokenizer-mode "${TOKENIZER_MODE}"
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --enable-auto-tool-choice
    --reasoning-parser "${REASONING_PARSER}"
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"
    --model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}"
    --quantization ascend
    --port "${PORT}"
    --block-size "${BLOCK_SIZE}"
    --speculative-config "${SPECULATIVE_CONFIG}"
    "${graph_args[@]}"
    --async-scheduling
    --additional-config "${ADDITIONAL_CONFIG}"
)

if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    serve_args+=(--enable-expert-parallel)
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    read -r -a extra_vllm_args <<< "${EXTRA_VLLM_ARGS}"
    serve_args+=("${extra_vllm_args[@]}")
fi

{
    echo "OUT_DIR=${OUT_DIR}"
    echo "DSV4_EXPERIMENT_MODE=${DSV4_EXPERIMENT_MODE}"
    echo "MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH}"
    printf 'vllm'
    printf ' %q' "${serve_args[@]}"
    printf '\n'
} | tee "${OUT_DIR}/serve_command.txt"

vllm "${serve_args[@]}" 2>&1 | tee "${OUT_DIR}/run_dsv4.log"
