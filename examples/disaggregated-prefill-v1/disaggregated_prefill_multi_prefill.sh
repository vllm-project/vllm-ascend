#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling We will
# launch 2 vllm instances (1 for prefill and 1 for decode), and then transfer
# the KV cache between them.

set -xe

current_dir=$(dirname "$0")

# vLLM Environment configuration
export VLLM_USE_V1=1

# vLLM-Ascend Environment configuration
export GLOBAL_RANKTABLE="${current_dir}/global_ranktable.json"
# The following environment variables are required for LLMDataDist.
export PROMPT_DEVICE_ID=0,1,2,3
export DECODE_DEVICE_ID=4,5,6,7
export TENSOR_PARALLEL_SIZE=$(($(echo $PROMPT_DEVICE_ID | grep -o ',' | wc -l) + 1))

# Model Configuration
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Generate the global rank table
if [ ! -f "${GLOBAL_RANKTABLE}" ]; then
    echo "Generating global rank table..."
    # TODO(jianzs): Impl a tool to generate the global rank table automatically
else
    echo "Global rank table already exists."
fi

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &>/dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi

# a function that waits vLLM server to start
wait_for_server() {
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

ASCEND_RT_VISIBLE_DEVICES=${PROMPT_DEVICE_ID} vllm serve ${MODEL_NAME} \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --kv-transfer-config \
    '{
        "kv_connector": "AscendHcclConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": 2,
        "kv_connector_extra_config": {
            "local_server_id": "server-0"
        }
    }' &

ASCEND_RT_VISIBLE_DEVICES=${DECODE_DEVICE_ID} vllm serve ${MODEL_NAME} \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --kv-transfer-config \
    '{
        "kv_connector": "AscendHcclConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "kv_consumer",
        "kv_rank": 1,
        "kv_parallel_size": 2,
        "kv_connector_extra_config": {
            "local_server_id": "server-1"
        }
    }' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

echo "🚧🚧 Warning: server started 🚧🚧"

python3 disagg_prefill_proxy_server.py
