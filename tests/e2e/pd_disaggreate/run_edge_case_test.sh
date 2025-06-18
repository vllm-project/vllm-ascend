#!/bin/bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export CLOSE_MATMUL_K_SHIFT=1
export VLLM_USE_V1=1

set -xe

# Models to run
MODELS=(
    ""Qwen/Qwen2.5-0.5B-Instruct""
)

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Gen ranktable
RANKTABLE_PATH=${GIT_ROOT}/examples/disaggregate_prefill_v1/ranktable.json
if [ -f "$RANKTABLE_PATH" ]; then
    rm "$RANKTABLE_PATH"
fi
cd ${GIT_ROOT}/examples/disaggregate_prefill_v1
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
bash gen_ranktable.sh --ips $LOCAL_HOST  --network-card-name enp189s0f0 --prefill-device-cnt 1 --decode-device-cnt 1
cd -
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH="$RANKTABLE_PATH"

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/health > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == *"deepseek"* ]]; then
    extra_args="--trust-remote-code"
  fi

  echo "$extra_args"
}


# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")
  
  # Start prefill instance
  PREFILL_PORT=8001

  BASE_CMD="ASCEND_RT_VISIBLE_DEVICES=0 VLLM_LLMDD_RPC_PORT=5559 vllm serve $model_name \
  --port $PREFILL_PORT \
  --seed 1024 \
  --enforce-eager \
  --disable-log-requests \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{\"kv_connector\":\"LLMDataDistCMgrConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_device\":\"npu\",\"kv_parallel_size\":\"1\",\"kv_port\":\"20001\",\"engine_id\":\"0\",\"kv_connector_module_path\":\"vllm_ascend.distributed.llmdatadist_c_mgr_connector\"}'"

  if [ -n "$model_args" ]; then
  FULL_CMD="$BASE_CMD $model_args"
  else
  FULL_CMD="$BASE_CMD"
  fi

  eval "$FULL_CMD &"

  # Start decode instance
  DECODE_PORT=8002

  # Build the command with or without model-specific args
  BASE_CMD="ASCEND_RT_VISIBLE_DEVICES=1 VLLM_LLMDD_RPC_PORT=6000 vllm serve $model_name \
  --port $DECODE_PORT \
  --seed 1024 \
  --enforce-eager \
  --disable-log-requests \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{\"kv_connector\":\"LLMDataDistCMgrConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_device\":\"npu\",\"kv_parallel_size\":\"1\",\"kv_port\":\"20001\",\"engine_id\":\"0\",\"kv_connector_module_path\":\"vllm_ascend.distributed.llmdatadist_c_mgr_connector\"}'"

  if [ -n "$model_args" ]; then
  FULL_CMD="$BASE_CMD $model_args"
  else
  FULL_CMD="$BASE_CMD"
  fi

  eval "$FULL_CMD &"

  # Wait for all instances to start
  echo "Waiting for prefill instance on port $PORT to start..."
  wait_for_server $PREFILL_PORT
  echo "Waiting for decode instance on port $PORT to start..."
  wait_for_server $DECODE_PORT

  # Build the command for the proxy server with all the hosts and ports
  PROXY_PORT=8192
  PROXY_CMD="python ${GIT_ROOT}/examples/disaggregate_prefill_v1/toy_proxy_server.py --port $PROXY_PORT"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORT}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORT}"
  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  PREFILL_PORT=$PREFILL_PORT DECODE_PORT=$DECODE_PORT PROXY_PORT=$PROXY_PORT python -m pytest -s -v ${GIT_ROOT}/tests/e2e/pd_disaggreate/test_edge_cases.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"