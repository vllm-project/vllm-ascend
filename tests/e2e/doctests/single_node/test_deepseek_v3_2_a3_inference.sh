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

# Global environment variables from Single-node Deployment (Atlas A3)
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Start Server (Background)
echo "Starting vLLM server (Atlas A3 Configuration)..."
# Extracted from docs/source/tutorials/models/DeepSeek-V3.2.md
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
--host ${NODE_IP} \
--port ${PORT} \
--data-parallel-size 2 \
--tensor-parallel-size 8 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3_2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' &

SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Ensure server is killed on exit
trap 'kill $SERVER_PID' EXIT

# Wait for readiness
if wait_for_server $SERVER_PID; then
    # Inference Test Case 1: Functional Verification (curl)
    (
        CMD_1="curl http://${NODE_IP}:${PORT}/v1/completions \
            -H \"Content-Type: application/json\" \
            -d '{
                \"model\": \"deepseek_v3_2\",
                \"prompt\": \"The future of AI is\",
                \"max_completion_tokens\": 50,
                \"temperature\": 0
            }'"
        run_test_case "Functional Verification (curl)" "$CMD_1"
    )

    # Inference Test Case 2: vLLM Benchmark (Performance)
    (
        # Using the model path from single-node deployment as tokenizer
        export VLLM_USE_MODELSCOPE=true
        CMD_2="vllm bench serve --model deepseek_v3_2 \
        --tokenizer /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
        --dataset-name random \
        --random-input 200 \
        --num-prompts 200 \
        --request-rate 1 \
        --save-result \
        --result-dir ./ \
        --host ${NODE_IP} \
        --port ${PORT}"
        run_test_case "vLLM Benchmark (Performance)" "$CMD_2"
    )
else
    echo "Server failed to start. Skipping tests."
    exit 1
fi
