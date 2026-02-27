#!/bin/bash

# Function to run a test case and handle failure
run_test_case() {
    echo "Running: $1"
    # Capture output and exit code
    OUTPUT=$(eval "$2" 2>&1)
    EXIT_CODE=$?
    echo "$OUTPUT"

    # Check for specific failure patterns in output even if exit code is 0
    if [[ "$OUTPUT" == *"Failed requests"* ]] || [[ "$OUTPUT" == *"Error"* ]] || [[ "$OUTPUT" == *"404 Not Found"* ]]; then
        # Check if Failed requests is not 0
        if echo "$OUTPUT" | grep -q "Failed requests:\s*[1-9]"; then
             EXIT_CODE=1
        fi
        # Check for 404 Not Found in logs
        if echo "$OUTPUT" | grep -q "404 Not Found"; then
             EXIT_CODE=1
        fi
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Test Case '$1' PASSED"
    else
        echo "Test Case '$1' FAILED"
    fi
    echo "--------------------------------------------------"
}

# Function to wait for server
wait_for_server() {
    local pid=$1
    echo "Waiting for server to start at http://${NODE_IP}:${PORT}..."
    local retries=0
    local max_retries=120 # 20 minutes (10s * 120)
    while ! curl -s http://${NODE_IP}:${PORT}/health > /dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "Server process $pid has terminated unexpectedly."
            return 1
        fi
        sleep 10
        retries=$((retries+1))
        if [ $retries -ge $max_retries ]; then
            echo "Server failed to start within timeout."
            return 1
        fi
        echo "Waiting... ($retries/$max_retries)"
    done
    echo "Server is ready!"
    return 0
}

# Set default IP and Port
NODE_IP="${NODE_IP:-127.0.0.1}"
PORT="${PORT:-8000}"

# Global environment variables from Single Node A3 (64G*16) Performance section
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export TASK_QUEUE_ENABLE=1
# Fix for user-specified max_model_len > derived max_model_len
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Start Server (Background)
echo "Starting vLLM server (Atlas A3 Configuration)..."
# Extracted from docs/source/tutorials/models/Qwen3-235B-A22B.md (Single Node A3 Performance)
vllm serve vllm-ascend/Qwen3-235B-A22B-W8A8 \
--host ${NODE_IP} \
--port ${PORT} \
--tensor-parallel-size 4 \
--data-parallel-size 4 \
--seed 1024 \
--quantization ascend \
--served-model-name qwen3 \
--max-num-seqs 128 \
--max-model-len 40960 \
--max-num-batched-tokens 16384 \
--enable-expert-parallel \
--trust-remote-code \
--gpu-memory-utilization 0.9 \
--no-enable-prefix-caching \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--async-scheduling &

SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Ensure server is killed on exit
cleanup() {
    echo "Stopping server with PID $SERVER_PID..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}
trap cleanup EXIT

# Wait for readiness
if wait_for_server $SERVER_PID; then
    # Inference Test Case 1: Functional Verification (curl)
    (
        CMD_1="curl http://${NODE_IP}:${PORT}/v1/completions \
            -H \"Content-Type: application/json\" \
            -d '{
                \"model\": \"qwen3\",
                \"prompt\": \"The future of AI is\",
                \"max_completion_tokens\": 50,
                \"temperature\": 0
            }'"
        run_test_case "Functional Verification (curl)" "$CMD_1"
    )

    # Inference Test Case 2: vLLM Benchmark (Single Node A3 Performance)
    (
        CMD_2="vllm bench serve --model qwen3 \
        --tokenizer vllm-ascend/Qwen3-235B-A22B-W8A8 \
        --ignore-eos \
        --dataset-name random \
        --random-input-len 3584 \
        --random-output-len 1536 \
        --num-prompts 8 \
        --max-concurrency 160 \
        --request-rate 24 \
        --host ${NODE_IP} \
        --port ${PORT}"
        run_test_case "vLLM Benchmark (Single Node A3 Performance)" "$CMD_2"
    )
else
    echo "Server failed to start. Skipping tests."
    exit 1
fi
