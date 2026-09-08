#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Reproducible baseline/typed A/B for a dedicated, otherwise idle NPU die.
# The caller is responsible for checking ownership and HBM availability first.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local model directory}
RESULT_ROOT=${RESULT_ROOT:-${SCRIPT_DIR}/results/typed-e2e}
PORT=${PORT:-18080}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
FIXED_QPS=${FIXED_QPS:-2}
NUM_PROMPTS=${NUM_PROMPTS:-128}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS:-"8 16 32 48 64 80 96 128"}
SERVED_MODEL_NAME=qwen35-jenga
SEED=20260901
GDN_DECODE_BACKEND=${GDN_DECODE_BACKEND:-ascendc}

mkdir -p "${RESULT_ROOT}"
server_pid=""

stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
  server_pid=""
}
trap stop_server EXIT

wait_for_server() {
  local attempts=0
  until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
    attempts=$((attempts + 1))
    if [[ ${attempts} -ge 180 ]]; then
      return 1
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
}

run_mode() {
  local mode=$1
  local enabled=$2
  local mode_dir="${RESULT_ROOT}/${mode}"
  mkdir -p "${mode_dir}"

  VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=${enabled} \
  VLLM_ASCEND_GDN_DECODE_BACKEND="${GDN_DECODE_BACKEND}" \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --load-format dummy \
      --seed "${SEED}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs 256 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --skip-mm-profiling \
      --no-enable-prefix-caching \
      --enforce-eager \
      >"${mode_dir}/server.log" 2>&1 &
  server_pid=$!
  wait_for_server

  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Explain paged KV cache in one sentence.\",\"temperature\":0,\"max_tokens\":32,\"seed\":${SEED}}" \
    "http://127.0.0.1:${PORT}/v1/completions" \
    >"${mode_dir}/deterministic.json"

  vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/completions \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${FIXED_QPS}" \
    --max-concurrency 32 \
    --seed "${SEED}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --result-dir "${mode_dir}" \
    --result-filename fixed-qps.json

  for concurrency in ${CONCURRENCY_LEVELS}; do
    if ! vllm bench serve \
      --backend openai \
      --base-url "http://127.0.0.1:${PORT}" \
      --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" \
      --tokenizer "${MODEL_PATH}" \
      --dataset-name random \
      --random-input-len 4096 \
      --random-output-len 64 \
      --num-prompts $((concurrency * 2)) \
      --request-rate inf \
      --max-concurrency "${concurrency}" \
      --seed "${SEED}" \
      --save-result \
      --result-dir "${mode_dir}" \
      --result-filename "concurrency-${concurrency}.json"; then
      printf '%s\n' "${concurrency}" >"${mode_dir}/first-failed-concurrency.txt"
      break
    fi
  done

  stop_server
}

run_mode uniform 0
run_mode typed 1

python - "${RESULT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
baseline = json.loads((root / "uniform" / "deterministic.json").read_text())
typed = json.loads((root / "typed" / "deterministic.json").read_text())
baseline_text = baseline["choices"][0]["text"]
typed_text = typed["choices"][0]["text"]
result = {
    "exact_match": baseline_text == typed_text,
    "uniform_text": baseline_text,
    "typed_text": typed_text,
}
(root / "deterministic-comparison.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
if not result["exact_match"]:
    raise SystemExit("typed output differs from the uniform baseline")
PY
