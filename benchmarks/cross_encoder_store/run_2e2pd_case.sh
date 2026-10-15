#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CASE=${CASE:?Set CASE=p2p or CASE=store}
if [[ "$CASE" != "p2p" && "$CASE" != "store" ]]; then
  echo "CASE must be p2p or store" >&2
  exit 2
fi

MODEL=${MODEL:-Eco-Tech/Qwen3.6-35B-A3B-w8a8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the extracted prepared directory}
VLLM_ROOT=${VLLM_ROOT:-/vllm-workspace/vllm}
EPD_PROXY=${EPD_PROXY:-${VLLM_ROOT}/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py}
PYTHON_BIN=${PYTHON_BIN:-python3}
VLLM_BIN=${VLLM_BIN:-vllm}
ROUND=${ROUND:-1}
RESULT_ROOT=${RESULT_ROOT:-${PWD}/cross_encoder_store_results}
RUN_ID=${RUN_ID:-${CASE}-round${ROUND}-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RESULT_ROOT}/${RUN_ID}

NPU_E0=${NPU_E0:-0}
NPU_E1=${NPU_E1:-1}
NPU_PD0=${NPU_PD0:-2}
NPU_PD1=${NPU_PD1:-3}
E0_PORT=${E0_PORT:-18100}
E1_PORT=${E1_PORT:-18101}
PD0_PORT=${PD0_PORT:-18102}
PD1_PORT=${PD1_PORT:-18103}
PD0_RESERVATION_PORT=${PD0_RESERVATION_PORT:-19102}
PD1_RESERVATION_PORT=${PD1_RESERVATION_PORT:-19103}
PROXY_E0_PD0_PORT=${PROXY_E0_PD0_PORT:-18200}
PROXY_E0_PD1_PORT=${PROXY_E0_PD1_PORT:-18201}
PROXY_E1_PD0_PORT=${PROXY_E1_PD0_PORT:-18210}
PROXY_E1_PD1_PORT=${PROXY_E1_PD1_PORT:-18211}
IMAGE_PORT=${IMAGE_PORT:-18300}
MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50051}
STORE_MODE=${STORE_MODE:-embedded}
STORE_CLIENT_HOST=${STORE_CLIENT_HOST:-}
if [[ -z "$STORE_CLIENT_HOST" ]]; then
  STORE_CLIENT_HOST=$("$PYTHON_BIN" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
fi
STORE_CLIENT_PORT=${STORE_CLIENT_PORT:-50052}
STORE_CLIENT_HIXL_PORT=${STORE_CLIENT_HIXL_PORT:-16667}
STORE_CLIENT_THREADS=${STORE_CLIENT_THREADS:-8}
STORE_CLIENT_FABRIC_CAPACITY_GB=${STORE_CLIENT_FABRIC_CAPACITY_GB:-20}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-98304}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-96}
POPULATION_CONCURRENCY=${POPULATION_CONCURRENCY:-1}
MEASUREMENT_CONCURRENCY=${MEASUREMENT_CONCURRENCY:-96}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-16}
POPULATION_CLIENT_PROCESSES=${POPULATION_CLIENT_PROCESSES:-1}
MEASUREMENT_CLIENT_PROCESSES=${MEASUREMENT_CLIENT_PROCESSES:-8}
WARMUP_CLIENT_PROCESSES=${WARMUP_CLIENT_PROCESSES:-8}
CLIENT_MAX_TOKENS=${CLIENT_MAX_TOKENS:-32}
CLIENT_REQUEST_TIMEOUT=${CLIENT_REQUEST_TIMEOUT:-1800}
STORE_GLOBAL_SEGMENT_SIZE=${STORE_GLOBAL_SEGMENT_SIZE:-8GB}
STORE_STANDALONE_GLOBAL_SEGMENT_SIZE=${STORE_STANDALONE_GLOBAL_SEGMENT_SIZE:-16GB}
STORE_LOCAL_BUFFER_SIZE=${STORE_LOCAL_BUFFER_SIZE:-512MB}
# A3 Store segments must use FabricMem (or an explicitly selected RoCE path)
# so a peer Encoder can directly address the remote Host allocation. Keep the
# environment identical in the p2p and store cases for a stable A/B setup.
ASCEND_ENABLE_USE_FABRIC_MEM=${ASCEND_ENABLE_USE_FABRIC_MEM:-1}
PRODUCER_STAGING_BYTES=${PRODUCER_STAGING_BYTES:-3221225472}
PRODUCER_BOUNCE_BYTES=${PRODUCER_BOUNCE_BYTES:-268435456}
CONSUMER_BUFFER_BYTES=${CONSUMER_BUFFER_BYTES:-3221225472}
START_TIMEOUT_SECONDS=${START_TIMEOUT_SECONDS:-3600}
STORE_DRAIN_TIMEOUT_SECONDS=${STORE_DRAIN_TIMEOUT_SECONDS:-1800}
STORE_VERIFY_TIMEOUT_SECONDS=${STORE_VERIFY_TIMEOUT_SECONDS:-60}

if [[ "$ASCEND_ENABLE_USE_FABRIC_MEM" != "0" && \
      "$ASCEND_ENABLE_USE_FABRIC_MEM" != "1" ]]; then
  echo "ASCEND_ENABLE_USE_FABRIC_MEM must be 0 or 1" >&2
  exit 2
fi
if [[ "$STORE_MODE" != "embedded" && "$STORE_MODE" != "standalone-store" ]]; then
  echo "STORE_MODE must be embedded or standalone-store" >&2
  exit 2
fi
if [[ "$CASE" == "store" && "$ASCEND_ENABLE_USE_FABRIC_MEM" != "1" && \
      "${HCCL_INTRA_ROCE_ENABLE:-0}" != "1" ]]; then
  echo "store requires FabricMem or an explicitly selected RoCE path" >&2
  exit 2
fi
# ASCEND_BUFFER_POOL is deprecated in the matching vLLM Ascend release. Do not
# accidentally combine an inherited legacy intermediate pool with FabricMem.
unset ASCEND_BUFFER_POOL
export ASCEND_ENABLE_USE_FABRIC_MEM

mkdir -p "$RUN_DIR"
PIDS=()
CLIENT_PID=

log_progress() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

show_failure() {
  local label=$1 log=$2
  log_progress "[ERROR][${label}] process failed; last 80 log lines follow"
  if [[ -s "$log" ]]; then
    tail -n 80 "$log" | sed "s/^/[${label}] /"
  else
    log_progress "[ERROR][${label}] log is missing or empty: ${log}"
  fi
}

show_representative_status() {
  local label=$1 log=$2 line
  line=$(grep -E 'Loading|loading|Ready|ready|Starting|Graph|profile|memory|ERROR|Error|Traceback' \
    "$log" 2>/dev/null | tail -n 1 || true)
  if [[ -n "$line" ]]; then
    printf '[%(%Y-%m-%d %H:%M:%S)T] [%s] %.240s\n' -1 "$label" "$line"
  else
    log_progress "[${label}] still starting; detailed log: ${log}"
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$CLIENT_PID" ]]; then
    kill "$CLIENT_PID" 2>/dev/null || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_http() {
  local label=$1 port=$2 pid=$3 log=$4
  local path=${5:-/health}
  local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
  local next_report=$((SECONDS + 30))
  log_progress "[WAIT][${label}] http://127.0.0.1:${port}${path}"
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      show_failure "$label" "$log"
      return 1
    fi
    if curl --max-time 2 -fsS "http://127.0.0.1:${port}${path}" >/dev/null 2>&1; then
      log_progress "[READY][${label}] port=${port}"
      return 0
    fi
    if ((SECONDS >= next_report)); then
      show_representative_status "$label" "$log"
      next_report=$((SECONDS + 30))
    fi
    sleep 2
  done
  show_failure "$label" "$log"
  return 1
}

wait_for_tcp() {
  local label=$1 port=$2 pid=$3 log=$4
  local host=${5:-127.0.0.1}
  local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
  log_progress "[WAIT][${label}] tcp://${host}:${port}"
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      show_failure "$label" "$log"
      return 1
    fi
    if "$PYTHON_BIN" -c \
      'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),2); s.close()' \
      "$host" "$port" 2>/dev/null; then
      log_progress "[READY][${label}] port=${port}"
      return 0
    fi
    sleep 2
  done
  show_failure "$label" "$log"
  return 1
}

start_logged() {
  local log=$1
  shift
  "$@" >"$log" 2>&1 &
  PIDS+=("$!")
}

run_phase() {
  local phase=$1 manifest=$2 concurrency=$3 client_processes=$4 output=$5 expected=$6
  log_progress "[PHASE][${phase}] start requests=${expected} concurrency=${concurrency} client_processes=${client_processes}"
  "${CLIENT_COMMON[@]}" \
    --manifest "$manifest" \
    --phase "$phase" \
    --concurrency "$concurrency" \
    --client-processes "$client_processes" \
    --max-tokens "$CLIENT_MAX_TOKENS" \
    --request-timeout "$CLIENT_REQUEST_TIMEOUT" \
    --output "$output" &
  CLIENT_PID=$!

  while kill -0 "$CLIENT_PID" 2>/dev/null; do
    sleep 10
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
      break
    fi
    local observed
    observed=$(
      { grep -hEo "ces-${phase}-[0-9]{4}-[0-9]+" \
          "${RUN_DIR}"/proxy_*.log 2>/dev/null || true; } |
        sort -u |
        wc -l
    )
    log_progress "[PROGRESS][${phase}] observed=${observed}/${expected}"
  done

  local status=0
  wait "$CLIENT_PID" || status=$?
  CLIENT_PID=
  if ((status != 0)); then
    log_progress "[ERROR][${phase}] client exited with status ${status}"
    return "$status"
  fi
  log_progress "[PHASE][${phase}] complete output=${output}"
}

PRODUCER_EXTRA=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps({"mooncake_protocol":"ascend","ascend_mooncake_bounce_arena_size":int(sys.argv[1])}))' "$PRODUCER_BOUNCE_BYTES")
if [[ "$CASE" == "store" ]]; then
  PRODUCER_EXTRA=$("$PYTHON_BIN" -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(cross_encoder_cache=True, embedding_cache_prefix=sys.argv[2]); print(json.dumps(d))' "$PRODUCER_EXTRA" "$RUN_ID")
fi
PRODUCER_CONFIG=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps({"ec_connector":"ECMooncakeConnector","ec_role":"ec_producer","ec_buffer_size":int(sys.argv[1]),"ec_buffer_device":"npu","ec_connector_extra_config":json.loads(sys.argv[2])}))' "$PRODUCER_STAGING_BYTES" "$PRODUCER_EXTRA")
consumer_config() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.dumps({"ec_connector":"ECMooncakeConnector","ec_role":"ec_consumer","ec_ip":"127.0.0.1","ec_port":int(sys.argv[1]),"ec_buffer_size":int(sys.argv[2]),"ec_buffer_device":"npu","ec_connector_extra_config":{"mooncake_protocol":"ascend"}}))' "$1" "$CONSUMER_BUFFER_BYTES"
}
PD0_CONFIG=$(consumer_config "$PD0_RESERVATION_PORT")
PD1_CONFIG=$(consumer_config "$PD1_RESERVATION_PORT")

if [[ "$CASE" == "store" ]]; then
  STORE_CONFIG=${RUN_DIR}/mooncake_store.json
  STORE_REQUESTER_GLOBAL_SEGMENT_SIZE=$STORE_GLOBAL_SEGMENT_SIZE
  if [[ "$STORE_MODE" == "standalone-store" ]]; then
    STORE_REQUESTER_GLOBAL_SEGMENT_SIZE=0
  fi
  "$PYTHON_BIN" -c 'import json,sys; json.dump({"mode":sys.argv[2],"metadata_server":"P2PHANDSHAKE","master_server_address":f"127.0.0.1:{sys.argv[3]}","global_segment_size":sys.argv[4],"local_buffer_size":sys.argv[5],"protocol":"ascend","device_name":"","enable_offload":False},open(sys.argv[1],"w"),indent=2)' "$STORE_CONFIG" "$STORE_MODE" "$MOONCAKE_MASTER_PORT" "$STORE_REQUESTER_GLOBAL_SEGMENT_SIZE" "$STORE_LOCAL_BUFFER_SIZE"
  log_progress "[START][STORE] Mooncake master port=${MOONCAKE_MASTER_PORT}"
  start_logged "${RUN_DIR}/mooncake_master.log" mooncake_master --port "$MOONCAKE_MASTER_PORT"
  MASTER_PID=${PIDS[${#PIDS[@]}-1]}
  wait_for_tcp "STORE" "$MOONCAKE_MASTER_PORT" "$MASTER_PID" "${RUN_DIR}/mooncake_master.log"

  if [[ "$STORE_MODE" == "standalone-store" ]]; then
    STORE_CLIENT_NPUS="${NPU_E0},${NPU_E1},${NPU_PD0},${NPU_PD1}"
    STORE_CLIENT_RESOURCE_CONFIG=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps({"comm_resource_config":{"listen_port":int(sys.argv[1])},"fabric_memory.max_capacity":int(sys.argv[2])}))' "$STORE_CLIENT_HIXL_PORT" "$STORE_CLIENT_FABRIC_CAPACITY_GB")
    log_progress "[START][STORE-CLIENT] host=${STORE_CLIENT_HOST} port=${STORE_CLIENT_PORT} npus=${STORE_CLIENT_NPUS} global_segment=${STORE_STANDALONE_GLOBAL_SEGMENT_SIZE}"
    start_logged "${RUN_DIR}/mooncake_client.log" env \
      ASCEND_RT_VISIBLE_DEVICES="$STORE_CLIENT_NPUS" \
      ASCEND_ENABLE_USE_FABRIC_MEM="$ASCEND_ENABLE_USE_FABRIC_MEM" \
      ASCEND_GLOBAL_RESOURCE_CONFIG="$STORE_CLIENT_RESOURCE_CONFIG" \
      mooncake_client \
      --host="$STORE_CLIENT_HOST" \
      --port="$STORE_CLIENT_PORT" \
      --master_server_address="127.0.0.1:${MOONCAKE_MASTER_PORT}" \
      --metadata_server=P2PHANDSHAKE \
      --protocol=ascend \
      --global_segment_size="$STORE_STANDALONE_GLOBAL_SEGMENT_SIZE" \
      --local_buffer_size="$STORE_LOCAL_BUFFER_SIZE" \
      --enable_offload=false \
      --start_offload_rpc_server=false \
      --tenant_id=default \
      --threads="$STORE_CLIENT_THREADS"
    STORE_CLIENT_PID=${PIDS[${#PIDS[@]}-1]}
    wait_for_tcp "STORE-CLIENT" "$STORE_CLIENT_PORT" "$STORE_CLIENT_PID" \
      "${RUN_DIR}/mooncake_client.log" "$STORE_CLIENT_HOST"
  fi
fi

COMMON_ARGS=(
  serve "$MODEL"
  --host 127.0.0.1
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --quantization ascend
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --limit-mm-per-prompt '{"image":32,"video":0}'
  --mm-tensor-ipc direct_rpc
  --mm-processor-device cpu
)

start_encoder() {
  local npu=$1 port=$2 log=$3
  if [[ "$CASE" == "store" ]]; then
    start_logged "$log" env \
      ASCEND_RT_VISIBLE_DEVICES="$npu" \
      ASCEND_ENABLE_USE_FABRIC_MEM="$ASCEND_ENABLE_USE_FABRIC_MEM" \
      VLLM_USE_V2_MODEL_RUNNER=1 \
      MOONCAKE_EC_PROTOCOL=ascend \
      MOONCAKE_CONFIG_PATH="$STORE_CONFIG" \
      "$VLLM_BIN" "${COMMON_ARGS[@]}" \
      --port "$port" \
      --gpu-memory-utilization 0.90 \
      --enable-request-id-headers \
      --ec-transfer-config "$PRODUCER_CONFIG"
  else
    start_logged "$log" env \
      ASCEND_RT_VISIBLE_DEVICES="$npu" \
      ASCEND_ENABLE_USE_FABRIC_MEM="$ASCEND_ENABLE_USE_FABRIC_MEM" \
      VLLM_USE_V2_MODEL_RUNNER=1 \
      MOONCAKE_EC_PROTOCOL=ascend \
      "$VLLM_BIN" "${COMMON_ARGS[@]}" \
      --port "$port" \
      --gpu-memory-utilization 0.90 \
      --enable-request-id-headers \
      --ec-transfer-config "$PRODUCER_CONFIG"
  fi
}

start_pd() {
  local npu=$1 port=$2 reservation_port=$3 config=$4 log=$5
  start_logged "$log" env \
    ASCEND_RT_VISIBLE_DEVICES="$npu" \
    ASCEND_ENABLE_USE_FABRIC_MEM="$ASCEND_ENABLE_USE_FABRIC_MEM" \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    MOONCAKE_EC_PROTOCOL=ascend \
    EC_MOONCAKE_RESERVATION_PORT="$reservation_port" \
    "$VLLM_BIN" "${COMMON_ARGS[@]}" \
    --port "$port" \
    --gpu-memory-utilization 0.93 \
    --enable-mm-embeds \
    --enable-request-id-headers \
    --ec-transfer-config "$config"
}

log_progress "[RUN] case=${CASE} round=${ROUND} run_dir=${RUN_DIR}"
log_progress "[CONFIG][ASCEND] use_fabric_mem=${ASCEND_ENABLE_USE_FABRIC_MEM} hccl_intra_roce=${HCCL_INTRA_ROCE_ENABLE:-0} buffer_pool=unset"
log_progress "[START][E0] logical_npu=${NPU_E0} port=${E0_PORT}"
start_encoder "$NPU_E0" "$E0_PORT" "${RUN_DIR}/encoder_e0.log"
E0_PID=${PIDS[${#PIDS[@]}-1]}
log_progress "[START][E1] logical_npu=${NPU_E1} port=${E1_PORT}"
start_encoder "$NPU_E1" "$E1_PORT" "${RUN_DIR}/encoder_e1.log"
E1_PID=${PIDS[${#PIDS[@]}-1]}
log_progress "[START][PD0] logical_npu=${NPU_PD0} port=${PD0_PORT}"
start_pd "$NPU_PD0" "$PD0_PORT" "$PD0_RESERVATION_PORT" "$PD0_CONFIG" "${RUN_DIR}/pd0.log"
PD0_PID=${PIDS[${#PIDS[@]}-1]}
log_progress "[START][PD1] logical_npu=${NPU_PD1} port=${PD1_PORT}"
start_pd "$NPU_PD1" "$PD1_PORT" "$PD1_RESERVATION_PORT" "$PD1_CONFIG" "${RUN_DIR}/pd1.log"
PD1_PID=${PIDS[${#PIDS[@]}-1]}

wait_for_http "E0" "$E0_PORT" "$E0_PID" "${RUN_DIR}/encoder_e0.log"
wait_for_http "E1" "$E1_PORT" "$E1_PID" "${RUN_DIR}/encoder_e1.log"
wait_for_http "PD0" "$PD0_PORT" "$PD0_PID" "${RUN_DIR}/pd0.log"
wait_for_http "PD1" "$PD1_PORT" "$PD1_PID" "${RUN_DIR}/pd1.log"

start_proxy() {
  local encoder_port=$1 pd_port=$2 reservation_port=$3 proxy_port=$4 log=$5
  start_logged "$log" "$PYTHON_BIN" "$EPD_PROXY" \
    --host 127.0.0.1 \
    --port "$proxy_port" \
    --encode-servers-urls "http://127.0.0.1:${encoder_port}" \
    --prefill-servers-urls disable \
    --decode-servers-urls "http://127.0.0.1:${pd_port}" \
    --ec-consumer-zmq-addrs "tcp://127.0.0.1:${reservation_port}" \
    --decode-retries 0
}
start_proxy "$E0_PORT" "$PD0_PORT" "$PD0_RESERVATION_PORT" "$PROXY_E0_PD0_PORT" "${RUN_DIR}/proxy_e0_pd0.log"
P00_PID=${PIDS[${#PIDS[@]}-1]}
start_proxy "$E0_PORT" "$PD1_PORT" "$PD1_RESERVATION_PORT" "$PROXY_E0_PD1_PORT" "${RUN_DIR}/proxy_e0_pd1.log"
P01_PID=${PIDS[${#PIDS[@]}-1]}
start_proxy "$E1_PORT" "$PD0_PORT" "$PD0_RESERVATION_PORT" "$PROXY_E1_PD0_PORT" "${RUN_DIR}/proxy_e1_pd0.log"
P10_PID=${PIDS[${#PIDS[@]}-1]}
start_proxy "$E1_PORT" "$PD1_PORT" "$PD1_RESERVATION_PORT" "$PROXY_E1_PD1_PORT" "${RUN_DIR}/proxy_e1_pd1.log"
P11_PID=${PIDS[${#PIDS[@]}-1]}

log_progress "[START][PROXY] four fixed E/PD routes"
wait_for_http "P00" "$PROXY_E0_PD0_PORT" "$P00_PID" "${RUN_DIR}/proxy_e0_pd0.log"
wait_for_http "P01" "$PROXY_E0_PD1_PORT" "$P01_PID" "${RUN_DIR}/proxy_e0_pd1.log"
wait_for_http "P10" "$PROXY_E1_PD0_PORT" "$P10_PID" "${RUN_DIR}/proxy_e1_pd0.log"
wait_for_http "P11" "$PROXY_E1_PD1_PORT" "$P11_PID" "${RUN_DIR}/proxy_e1_pd1.log"

start_logged "${RUN_DIR}/image_server.log" "$PYTHON_BIN" -m http.server "$IMAGE_PORT" --bind 127.0.0.1 --directory "$DATA_DIR"
IMAGE_PID=${PIDS[${#PIDS[@]}-1]}
wait_for_http "IMAGE" "$IMAGE_PORT" "$IMAGE_PID" "${RUN_DIR}/image_server.log" "/"

CLIENT_COMMON=(
  "$PYTHON_BIN" "${SCRIPT_DIR}/benchmark_client.py"
  --image-base-url "http://127.0.0.1:${IMAGE_PORT}"
  --served-model-name "$SERVED_MODEL_NAME"
  --e0-pd0-url "http://127.0.0.1:${PROXY_E0_PD0_PORT}"
  --e0-pd1-url "http://127.0.0.1:${PROXY_E0_PD1_PORT}"
  --e1-pd0-url "http://127.0.0.1:${PROXY_E1_PD0_PORT}"
  --e1-pd1-url "http://127.0.0.1:${PROXY_E1_PD1_PORT}"
)

run_phase warmup "${DATA_DIR}/warmup.jsonl" "$WARMUP_CONCURRENCY" "$WARMUP_CLIENT_PROCESSES" \
  "${RUN_DIR}/warmup.json" 16

run_phase population "${DATA_DIR}/measurement.jsonl" "$POPULATION_CONCURRENCY" "$POPULATION_CLIENT_PROCESSES" \
  "${RUN_DIR}/population.json" 512

ENCODER_LOGS=("${RUN_DIR}/encoder_e0.log" "${RUN_DIR}/encoder_e1.log")
EXPECTED_IMAGES=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["measurement"]["unique_images"])' "${DATA_DIR}/metadata.json")
if [[ "$CASE" == "store" ]]; then
  log_progress "[STORE] waiting for exactly ${EXPECTED_IMAGES} successful PUT markers"
  "$PYTHON_BIN" "${SCRIPT_DIR}/store_log_check.py" \
    --logs "${ENCODER_LOGS[@]}" \
    --expected-puts "$EXPECTED_IMAGES" \
    --timeout "$STORE_DRAIN_TIMEOUT_SECONDS"
  log_progress "[STORE] population PUT verification complete"
fi

run_phase measurement "${DATA_DIR}/measurement.jsonl" "$MEASUREMENT_CONCURRENCY" "$MEASUREMENT_CLIENT_PROCESSES" \
  "${RUN_DIR}/measurement.json" 512

if [[ "$CASE" == "store" ]]; then
  log_progress "[STORE] verifying exactly ${EXPECTED_IMAGES} PUT and GET markers"
  "$PYTHON_BIN" "${SCRIPT_DIR}/store_log_check.py" \
    --logs "${ENCODER_LOGS[@]}" \
    --expected-puts "$EXPECTED_IMAGES" \
    --expected-gets "$EXPECTED_IMAGES" \
    --timeout "$STORE_VERIFY_TIMEOUT_SECONDS"
  log_progress "[STORE] measurement Store verification complete"
fi

log_progress "[DONE] case=${CASE} round=${ROUND} run_dir=${RUN_DIR}"
