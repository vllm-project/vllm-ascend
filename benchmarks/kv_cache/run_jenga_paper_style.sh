#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Scaled reproduction of the Jenga serving experiments on one dedicated A3 die.
# Run once for static_partition and once for address_table.  The caller must
# verify that the selected physical die is idle and must not pre-empt workloads.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

MODE=${MODE:?set MODE to static_partition or address_table}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local model directory}
SOURCE_DATASET=${SOURCE_DATASET:-${SCRIPT_DIR}/datasets/longbench_qwen35_512_2k_8k_32k.jsonl}
RESULT_ROOT=${RESULT_ROOT:-${SCRIPT_DIR}/results/jenga-paper-style}
WORKLOAD_DIR=${WORKLOAD_DIR:-${RESULT_ROOT}/workloads}
MODE_DIR=${RESULT_ROOT}/${MODE}
PORT=${PORT:-18085}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-27b}
SEED=${SEED:-20260901}
NUM_PROMPTS=${NUM_PROMPTS:-16}
MIXED_PROMPTS=${MIXED_PROMPTS:-16}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-32}
RATE_SWEEP=${RATE_SWEEP:-"0.2 0.4 0.8 1.6 3.2"}
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
GDN_DECODE_BACKEND=${GDN_DECODE_BACKEND:-ascendc}

mkdir -p "${MODE_DIR}"
python benchmarks/kv_cache/prepare_jenga_paper_workloads.py \
  --input "${SOURCE_DATASET}" \
  --output-dir "${WORKLOAD_DIR}" \
  --tokenizer-json "${MODEL_PATH}/tokenizer.json" \
  --max-model-len 32768 \
  --seed "${SEED}" \
  --per-bucket "${NUM_PROMPTS}" \
  --output-tokens "${OUTPUT_TOKENS}"

server_pid=""
sampler_pid=""

cleanup() {
  if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" || true
    wait "${sampler_pid}" || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" || true
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  for _ in $(seq 1 240); do
    if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

sample_metrics() {
  local output=$1
  (
    while true; do
      printf 'timestamp_seconds %.3f\n' "$(date +%s.%3N)"
      curl --silent "http://127.0.0.1:${PORT}/metrics" \
        | grep -E '^(vllm:|vllm_)' \
        | grep -E 'num_requests_running|num_requests_waiting|kv_cache_usage|cache_usage|time_to_first_token|time_per_output_token' \
        || true
      sleep 0.5
    done
  ) >"${output}" 2>&1 &
  sampler_pid=$!
}

stop_sampler() {
  if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" || true
    wait "${sampler_pid}" || true
  fi
  sampler_pid=""
}

run_benchmark() {
  local name=$1
  local dataset=$2
  local requests=$3
  local request_rate=$4
  local max_concurrency=$5
  sample_metrics "${MODE_DIR}/${name}-metrics.txt"
  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/completions \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name custom \
    --dataset-path "${dataset}" \
    --skip-chat-template \
    --disable-shuffle \
    --num-prompts "${requests}" \
    --output-len "${OUTPUT_TOKENS}" \
    --request-rate "${request_rate}" \
    --burstiness 1 \
    --max-concurrency "${max_concurrency}" \
    --ignore-eos \
    --temperature 0 \
    --seed "${SEED}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --plot-timeline \
    --result-dir "${MODE_DIR}" \
    --result-filename "${name}.json" \
    >"${MODE_DIR}/${name}.log" 2>&1
  stop_sampler
}

VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=1 \
VLLM_ASCEND_TYPED_KV_CACHE_MODE="${MODE}" \
VLLM_ASCEND_GDN_DECODE_BACKEND="${GDN_DECODE_BACKEND}" \
VLLM_BATCH_INVARIANT="${BATCH_INVARIANT}" \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  vllm serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --generation-config vllm \
    --seed "${SEED}" \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --skip-mm-profiling \
    --no-enable-prefix-caching \
    --enforce-eager \
    >"${MODE_DIR}/server.log" 2>&1 &
server_pid=$!
wait_for_server

# Deterministic correctness gate.  The analysis step requires byte-identical
# text between both allocation modes before reporting any A/B performance.
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Explain why KV cache pages improve serving efficiency.\",\"temperature\":0,\"max_tokens\":32,\"seed\":${SEED}}" \
  "http://127.0.0.1:${PORT}/v1/completions" \
  >"${MODE_DIR}/correctness-gate.json"
python - "${MODE_DIR}/correctness-gate.json" <<'PY'
import json
import pathlib
import sys

response = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
text = response["choices"][0]["text"]
if not text.strip() or len(set(text.strip())) < 8:
    raise SystemExit("correctness gate produced empty or degenerate text")
PY

# Figure-14 style burst throughput and decode/running-batch timeline.
run_benchmark \
  throughput-8k \
  "${WORKLOAD_DIR}/longbench-8192.jsonl" \
  "${NUM_PROMPTS}" \
  inf \
  "${NUM_PROMPTS}"

# Heterogeneous trace for memory occupancy and queueing behavior.
run_benchmark \
  mixed-memory \
  "${WORKLOAD_DIR}/longbench-mixed.jsonl" \
  "${MIXED_PROMPTS}" \
  inf \
  "${MIXED_PROMPTS}"

# Figure-15 style Poisson arrival sweep.
for rate in ${RATE_SWEEP}; do
  run_benchmark \
    "rate-${rate}" \
    "${WORKLOAD_DIR}/longbench-2048.jsonl" \
    "${NUM_PROMPTS}" \
    "${rate}" \
    "${NUM_PROMPTS}"
done

cleanup
trap - EXIT
