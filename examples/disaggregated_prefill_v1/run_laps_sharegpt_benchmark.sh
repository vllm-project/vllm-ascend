#!/usr/bin/env bash
set -euo pipefail

# Run a controlled LAPS benchmark on a 1P1D vLLM-Ascend deployment.
# Each case restarts Prefill, Decode, and the proxy, warms up once, then
# runs the measured ShareGPT workload with an open-loop request rate.

MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ds-r1-1p5b}"
DATASET_PATH="${DATASET_PATH:-/vllm-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
PROXY_DIR="${PROXY_DIR:-${VLLM_ASCEND_DIR}/examples/disaggregated_prefill_v1}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/laps_sharegpt_$(date +%Y%m%d_%H%M%S)}"
CASE_SUMMARY_CSV="case_summary.csv"
CLASS_SUMMARY_CSV="class_summary.csv"

PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3}"
DECODE_DEVICES="${DECODE_DEVICES:-4,5,6,7}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-13700}"
DECODE_PORT="${DECODE_PORT:-13701}"
PROXY_PORT="${PROXY_PORT:-8080}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"

NUM_PROMPTS="${NUM_PROMPTS:-5000}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-200}"
REQUEST_RATES="${REQUEST_RATES:-8 12}"
WARMUP_REQUEST_RATE="${WARMUP_REQUEST_RATE:-8}"
# Optional safety cap. Leave empty for a pure open-loop workload.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-600}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-5}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"

PREFILL_PID=""
DECODE_PID=""
PROXY_PID=""

mkdir -p "${RESULT_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

write_summary_headers() {
  cat >"${RESULT_DIR}/${CASE_SUMMARY_CSV}" <<'EOF'
case_name,label,variant,laps_enabled,laps_threshold,laps_wait_window_ms,laps_wait_max_batch,laps_long_prefill_cap,laps_short_reserved_ratio,request_rate,num_prompts,completed,failed,duration_s,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p99_itl_ms,mean_e2el_ms,median_e2el_ms,p99_e2el_ms
EOF
  cat >"${RESULT_DIR}/${CLASS_SUMMARY_CSV}" <<'EOF'
case_name,label,variant,laps_enabled,laps_threshold,laps_wait_window_ms,laps_wait_max_batch,laps_long_prefill_cap,laps_short_reserved_ratio,request_rate,class_name,class_threshold,num_requests,completed_requests,avg_input_len,avg_output_len,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_e2el_ms,median_e2el_ms,p99_e2el_ms
EOF
}

source_env() {
  # These files exist in the target container environment. Missing files are
  # tolerated so the script can still be linted or inspected elsewhere.
  set +u
  [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ] && source /usr/local/Ascend/cann-8.5.1/set_env.sh
  [ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
  [ -f /usr/local/Ascend/nnal/asdsip/set_env.sh ] && source /usr/local/Ascend/nnal/asdsip/set_env.sh
  set -euo pipefail
  export PYTHONPATH="${VLLM_DIR}:${PYTHONPATH:-}"
  export VLLM_USE_MODELSCOPE=False
  export HCCL_OP_EXPANSION_MODE=AIV
}

kill_tree() {
  local pid="$1"
  [ -z "${pid}" ] && return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  local children
  children="$(pgrep -P "${pid}" 2>/dev/null || true)"
  for child in ${children}; do
    kill_tree "${child}"
  done

  kill "${pid}" 2>/dev/null || true
}

wait_gone() {
  local pid="$1"
  local deadline=$((SECONDS + STOP_TIMEOUT_S))
  [ -z "${pid}" ] && return 0

  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Force killing pid=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
      break
    fi
    sleep 1
  done
}

kill_matching_cmd() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    log "Stopping existing process pid=${pid}, pattern=${pattern}"
    kill_tree "${pid}"
  done
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    wait_gone "${pid}"
  done
}

stop_existing_services() {
  kill_matching_cmd "vllm serve .*--port ${PREFILL_PORT}"
  kill_matching_cmd "vllm serve .*--port ${DECODE_PORT}"
  kill_matching_cmd "load_balance_proxy_server_example.py .*--port ${PROXY_PORT}"
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -Eq "[:.]${port}[[:space:]]"
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

print_port_users() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
  else
    log "Neither ss nor lsof is available to inspect port ${port}"
  fi
}

wait_port_free() {
  local port="$1"
  local name="$2"
  local deadline=$((SECONDS + PORT_FREE_TIMEOUT_S))

  while port_in_use "${port}"; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Port ${port} for ${name} is still in use after cleanup:"
      print_port_users "${port}"
      return 1
    fi
    sleep 1
  done
  log "Port ${port} for ${name} is free"
}

ensure_ports_free() {
  wait_port_free "${PREFILL_PORT}" "Prefill"
  wait_port_free "${DECODE_PORT}" "Decode"
  wait_port_free "${PROXY_PORT}" "Proxy"
}

stop_services() {
  log "Stopping previous services"
  kill_tree "${PROXY_PID}"
  kill_tree "${PREFILL_PID}"
  kill_tree "${DECODE_PID}"
  stop_existing_services
  wait_gone "${PROXY_PID}"
  wait_gone "${PREFILL_PID}"
  wait_gone "${DECODE_PID}"
  PROXY_PID=""
  PREFILL_PID=""
  DECODE_PID=""
  sleep "${SLEEP_AFTER_STOP_S}"
  ensure_ports_free
}

cleanup() {
  stop_services || true
}
trap cleanup EXIT

wait_http() {
  local url="$1"
  local name="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  log "Waiting for ${name}: ${url}"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Timed out waiting for ${name}"
      return 1
    fi
    sleep 2
  done
  log "${name} is ready"
}

wait_log() {
  local file="$1"
  local pattern="$2"
  local name="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  log "Waiting for ${name} log pattern: ${pattern}"
  until grep -q "${pattern}" "${file}" 2>/dev/null; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Timed out waiting for ${name}; last 80 log lines:"
      tail -80 "${file}" || true
      return 1
    fi
    sleep 2
  done
  log "${name} is ready"
}

start_prefill() {
  local case_name="$1"
  local laps_threshold="$2"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"

  (
    source_env
    cd "${VLLM_ASCEND_DIR}"
    export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES}"
    if [ "${laps_threshold}" = "off" ]; then
      unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH VLLM_ASCEND_LAPS_LONG_PREFILL_CAP VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
    else
      export VLLM_ASCEND_LAPS_SCHEDULING=1
      export VLLM_ASCEND_LAPS_THRESHOLD="${laps_threshold}"
      export VLLM_ASCEND_LAPS_WAIT_WINDOW_MS="${LAPS_WAIT_WINDOW_MS:-5}"
      export VLLM_ASCEND_LAPS_WAIT_MAX_BATCH="${LAPS_WAIT_MAX_BATCH:-4}"
      export VLLM_ASCEND_LAPS_LONG_PREFILL_CAP="${LAPS_LONG_PREFILL_CAP:-0}"
      export VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO="${LAPS_SHORT_RESERVED_RATIO:-0}"
      export VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-0}"
    fi
    exec vllm serve "${MODEL_PATH}" \
      --host "${PREFILL_HOST}" \
      --port "${PREFILL_PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --no-async-scheduling \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"engine_id\":\"0\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}},\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}}}"
  ) >"${log_file}" 2>&1 &
  PREFILL_PID=$!
  log "Started Prefill pid=${PREFILL_PID}, log=${log_file}"
}

start_decode() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_decode.log"

  (
    source_env
    cd "${VLLM_ASCEND_DIR}"
    export ASCEND_RT_VISIBLE_DEVICES="${DECODE_DEVICES}"
    unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH
    exec vllm serve "${MODEL_PATH}" \
      --host "${DECODE_HOST}" \
      --port "${DECODE_PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --no-async-scheduling \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"${DECODE_KV_PORT}\",\"engine_id\":\"1\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}},\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}}}"
  ) >"${log_file}" 2>&1 &
  DECODE_PID=$!
  log "Started Decode pid=${DECODE_PID}, log=${log_file}"
}

start_proxy() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_proxy.log"

  (
    source_env
    cd "${PROXY_DIR}"
    exec python3 load_balance_proxy_server_example.py \
      --host "${PROXY_HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-hosts "${PREFILL_HOST}" \
      --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${DECODE_HOST}" \
      --decoder-ports "${DECODE_PORT}"
  ) >"${log_file}" 2>&1 &
  PROXY_PID=$!
  log "Started Proxy pid=${PROXY_PID}, log=${log_file}"
}

run_bench() {
  local case_name="$1"
  local label="$2"
  local num_prompts="$3"
  local request_rate="$4"
  local result_file="$5"
  local log_file="${RESULT_DIR}/logs/${case_name}_${label}.log"
  local concurrency_args=()

  if [ -n "${MAX_CONCURRENCY}" ]; then
    concurrency_args=(--max-concurrency "${MAX_CONCURRENCY}")
  fi

  (
    source_env
    cd "${VLLM_DIR}"
    exec vllm bench serve \
      --backend vllm \
      --model "${SERVED_MODEL_NAME}" \
      --tokenizer "${MODEL_PATH}" \
      --endpoint /v1/completions \
      --dataset-name sharegpt \
      --dataset-path "${DATASET_PATH}" \
      --num-prompts "${num_prompts}" \
      --host "${PROXY_HOST}" \
      --port "${PROXY_PORT}" \
      --request-rate "${request_rate}" \
      "${concurrency_args[@]}" \
      --save-result \
      --save-detailed \
      --result-dir "${RESULT_DIR}" \
      --result-filename "${result_file}"
  ) 2>&1 | tee "${log_file}"
}

append_case_summaries() {
  local case_name="$1"
  local label="$2"
  local request_rate="$3"
  local laps_threshold="$4"
  local result_file="$5"
  local result_path="${RESULT_DIR}/${result_file}"

  local laps_enabled variant threshold_value wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio
  if [ "${laps_threshold}" = "off" ]; then
    laps_enabled=0
    variant="off"
    threshold_value=""
    wait_window_ms=""
    wait_max_batch=""
    long_prefill_cap=""
    short_reserved_ratio=""
  else
    laps_enabled=1
    variant="laps"
    threshold_value="${laps_threshold}"
    wait_window_ms="${LAPS_WAIT_WINDOW_MS:-5}"
    wait_max_batch="${LAPS_WAIT_MAX_BATCH:-4}"
    long_prefill_cap="${LAPS_LONG_PREFILL_CAP:-0}"
    short_reserved_ratio="${LAPS_SHORT_RESERVED_RATIO:-0}"
  fi

  python3 - "${result_path}" "${RESULT_DIR}/${CASE_SUMMARY_CSV}" "${RESULT_DIR}/${CLASS_SUMMARY_CSV}" \
    "${case_name}" "${label}" "${variant}" "${laps_enabled}" "${threshold_value}" \
    "${wait_window_ms}" "${wait_max_batch}" "${long_prefill_cap}" \
    "${short_reserved_ratio}" "${request_rate}" <<'PY'
import csv
import json
import math
import statistics
import sys


def percentile(values, q):
    if not values:
        return ""
    if len(values) == 1:
        return round(values[0] * 1000.0, 3)
    values = sorted(values)
    rank = (len(values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        value = values[lower]
    else:
        weight = rank - lower
        value = values[lower] * (1.0 - weight) + values[upper] * weight
    return round(value * 1000.0, 3)


def mean_ms(values):
    if not values:
        return ""
    return round(statistics.fmean(values) * 1000.0, 3)


def median_ms(values):
    if not values:
        return ""
    return round(statistics.median(values) * 1000.0, 3)


def mean_value(values):
    if not values:
        return ""
    return round(statistics.fmean(values), 3)


result_path, case_csv, class_csv, case_name, label, variant, laps_enabled, laps_threshold, wait_window_ms, wait_max_batch, long_prefill_cap, short_reserved_ratio, request_rate = sys.argv[1:]

with open(result_path, "r", encoding="utf-8") as f:
    data = json.load(f)

input_lens = data.get("input_lens", [])
output_lens = data.get("output_lens", [])
ttfts = data.get("ttfts", [])
itls = data.get("itls", [])
errors = data.get("errors", [])
threshold = int(laps_threshold) if laps_threshold else 256

rows = []
all_itls = []
for idx, input_len in enumerate(input_lens):
    output_len = output_lens[idx] if idx < len(output_lens) else 0
    ttft = ttfts[idx] if idx < len(ttfts) else None
    itl_values = itls[idx] if idx < len(itls) else []
    error = errors[idx] if idx < len(errors) else None
    success = output_len > 0 and ttft is not None
    e2el = None
    tpot = None
    if success:
      e2el = ttft + sum(itl_values or [])
      if output_len > 1:
        tpot = (e2el - ttft) / (output_len - 1)
      else:
        tpot = 0.0
      all_itls.extend(itl_values or [])
    rows.append(
        {
            "input_len": input_len,
            "output_len": output_len,
            "ttft": ttft,
            "e2el": e2el,
            "tpot": tpot,
            "itls": itl_values or [],
            "success": success and not error,
        }
    )

successful_rows = [row for row in rows if row["success"]]

case_row = {
    "case_name": case_name,
    "label": label,
    "variant": variant,
    "laps_enabled": laps_enabled,
    "laps_threshold": laps_threshold,
    "laps_wait_window_ms": wait_window_ms,
    "laps_wait_max_batch": wait_max_batch,
    "laps_long_prefill_cap": long_prefill_cap,
    "laps_short_reserved_ratio": short_reserved_ratio,
    "request_rate": request_rate,
    "num_prompts": len(rows),
    "completed": data.get("completed", len(successful_rows)),
    "failed": data.get("failed", len(rows) - len(successful_rows)),
    "duration_s": round(data.get("duration", 0.0), 3),
    "request_throughput": round(data.get("request_throughput", 0.0), 3),
    "output_throughput": round(data.get("output_throughput", 0.0), 3) if data.get("output_throughput") is not None else "",
    "total_token_throughput": round(data.get("total_token_throughput", 0.0), 3),
    "mean_ttft_ms": mean_ms([row["ttft"] for row in successful_rows if row["ttft"] is not None]),
    "median_ttft_ms": median_ms([row["ttft"] for row in successful_rows if row["ttft"] is not None]),
    "p99_ttft_ms": percentile([row["ttft"] for row in successful_rows if row["ttft"] is not None], 0.99),
    "mean_tpot_ms": mean_ms([row["tpot"] for row in successful_rows if row["tpot"] is not None]),
    "median_tpot_ms": median_ms([row["tpot"] for row in successful_rows if row["tpot"] is not None]),
    "p99_tpot_ms": percentile([row["tpot"] for row in successful_rows if row["tpot"] is not None], 0.99),
    "mean_itl_ms": mean_ms(all_itls),
    "median_itl_ms": median_ms(all_itls),
    "p99_itl_ms": percentile(all_itls, 0.99),
    "mean_e2el_ms": mean_ms([row["e2el"] for row in successful_rows if row["e2el"] is not None]),
    "median_e2el_ms": median_ms([row["e2el"] for row in successful_rows if row["e2el"] is not None]),
    "p99_e2el_ms": percentile([row["e2el"] for row in successful_rows if row["e2el"] is not None], 0.99),
}

with open(case_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(case_row.keys()))
    writer.writerow(case_row)

for class_name, predicate in (
    ("short", lambda row: row["input_len"] <= threshold),
    ("long", lambda row: row["input_len"] > threshold),
):
    class_rows = [row for row in rows if predicate(row)]
    successful_class_rows = [row for row in class_rows if row["success"]]
    class_row = {
        "case_name": case_name,
        "label": label,
        "variant": variant,
        "laps_enabled": laps_enabled,
        "laps_threshold": laps_threshold,
        "laps_wait_window_ms": wait_window_ms,
        "laps_wait_max_batch": wait_max_batch,
        "laps_long_prefill_cap": long_prefill_cap,
        "laps_short_reserved_ratio": short_reserved_ratio,
        "request_rate": request_rate,
        "class_name": class_name,
        "class_threshold": threshold,
        "num_requests": len(class_rows),
        "completed_requests": len(successful_class_rows),
        "avg_input_len": mean_value([row["input_len"] for row in class_rows]),
        "avg_output_len": mean_value([row["output_len"] for row in class_rows]),
        "mean_ttft_ms": mean_ms([row["ttft"] for row in successful_class_rows if row["ttft"] is not None]),
        "median_ttft_ms": median_ms([row["ttft"] for row in successful_class_rows if row["ttft"] is not None]),
        "p99_ttft_ms": percentile([row["ttft"] for row in successful_class_rows if row["ttft"] is not None], 0.99),
        "mean_tpot_ms": mean_ms([row["tpot"] for row in successful_class_rows if row["tpot"] is not None]),
        "median_tpot_ms": median_ms([row["tpot"] for row in successful_class_rows if row["tpot"] is not None]),
        "p99_tpot_ms": percentile([row["tpot"] for row in successful_class_rows if row["tpot"] is not None], 0.99),
        "mean_e2el_ms": mean_ms([row["e2el"] for row in successful_class_rows if row["e2el"] is not None]),
        "median_e2el_ms": median_ms([row["e2el"] for row in successful_class_rows if row["e2el"] is not None]),
        "p99_e2el_ms": percentile([row["e2el"] for row in successful_class_rows if row["e2el"] is not None], 0.99),
    }
    with open(class_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_row.keys()))
        writer.writerow(class_row)
PY
}

run_case() {
  local case_name="$1"
  local laps_threshold="$2"
  local request_rate="$3"

  log "========== CASE ${case_name} started =========="
  stop_services
  start_prefill "${case_name}" "${laps_threshold}"
  start_decode "${case_name}"

  wait_log "${RESULT_DIR}/logs/${case_name}_prefill.log" "Application startup complete" "Prefill startup"
  wait_log "${RESULT_DIR}/logs/${case_name}_decode.log" "Application startup complete" "Decode startup"
  wait_http "http://${PREFILL_HOST}:${PREFILL_PORT}/health" "Prefill"
  wait_http "http://${DECODE_HOST}:${DECODE_PORT}/health" "Decode"
  start_proxy "${case_name}"
  wait_log "${RESULT_DIR}/logs/${case_name}_proxy.log" "Application startup complete" "Proxy startup"
  wait_http "http://${PROXY_HOST}:${PROXY_PORT}/healthcheck" "Proxy"

  if [ "${laps_threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_name}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  log "Running warmup for ${case_name}: prompts=${WARMUP_PROMPTS}, request_rate=${WARMUP_REQUEST_RATE}"
  run_bench "${case_name}" "warmup" "${WARMUP_PROMPTS}" "${WARMUP_REQUEST_RATE}" "${case_name}_warmup.json"
  append_case_summaries "${case_name}" "warmup" "${WARMUP_REQUEST_RATE}" "${laps_threshold}" "${case_name}_warmup.json"

  log "Running measured benchmark for ${case_name}: prompts=${NUM_PROMPTS}, request_rate=${request_rate}"
  run_bench "${case_name}" "measured" "${NUM_PROMPTS}" "${request_rate}" "${case_name}.json"
  append_case_summaries "${case_name}" "measured" "${request_rate}" "${laps_threshold}" "${case_name}.json"

  log "========== CASE ${case_name} completed =========="
}

main() {
  source_env
  write_summary_headers
  log "Results will be written to ${RESULT_DIR}"
  log "Using open-loop request rates: ${REQUEST_RATES}"
  if [ -n "${MAX_CONCURRENCY}" ]; then
    log "Optional max concurrency cap enabled: ${MAX_CONCURRENCY}"
  else
    log "No max concurrency cap enabled"
  fi

  for request_rate in ${REQUEST_RATES}; do
    local rate_name="${request_rate//./p}"
    run_case "r${rate_name}_off" "off" "${request_rate}"
    run_case "r${rate_name}_laps_t256" "256" "${request_rate}"
    run_case "r${rate_name}_laps_t512" "512" "${request_rate}"
  done

  log "All benchmark cases completed. Results: ${RESULT_DIR}"
  log "Case summary CSV: ${RESULT_DIR}/${CASE_SUMMARY_CSV}"
  log "Class summary CSV: ${RESULT_DIR}/${CLASS_SUMMARY_CSV}"
}

main "$@"
