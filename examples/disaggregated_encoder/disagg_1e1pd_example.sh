#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p $LOG_PATH

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"

CARD_E="${CARD_E:-0}"
CARD_PD="${CARD_PD:-1}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout

NUM_PROMPTS="${NUM_PROMPTS:-100}"    # number of prompts to send in benchmark

###############################################################################
# Helpers
###############################################################################
# Find the git repository root directory
VLLM_ROOT="/vllm-workspace/vllm"

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
PD_LOG=$LOG_PATH/pd_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1   # prevent re-entrancy

    # Kill all tracked PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done

    # Wait a moment for graceful shutdown
    sleep 2

    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null

    echo "All processes stopped."
    exit 0
}

trap cleanup INT
trap cleanup USR1
trap cleanup TERM

# clear previous cache
echo "remove previous ec cache folder"
rm -rf $EC_SHARED_STORAGE_PATH

echo "make ec cache folder"
mkdir -p $EC_SHARED_STORAGE_PATH

###############################################################################
# Encoder worker
###############################################################################
ASCEND_RT_VISIBLE_DEVICES="$CARD_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.01 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    >"${ENC_LOG}" 2>&1 &

PIDS+=($!)

###############################################################################
# Prefill+Decode worker
###############################################################################
ASCEND_RT_VISIBLE_DEVICES="$CARD_PD" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.9 \
    --port "$PREFILL_DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    >"${PD_LOG}" 2>&1 &

PIDS+=($!)

# Wait for workers
wait_for_server $ENCODE_PORT
wait_for_server $PREFILL_DECODE_PORT

###############################################################################
# Proxy
###############################################################################
python ./disagg_epd_proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

PIDS+=($!)

wait_for_server $PROXY_PORT
echo "All services are up!"

###############################################################################
# Single request with local image
###############################################################################
echo "Running single request with local image (non-stream)..."
echo "Running single request with local image (non-stream)..."

base64_image=$(base64 -w 0 "${VLLM_ROOT}/tests/v1/ec_connector/integration/hato.jpg")

cat > /tmp/request.json << EOF
{
    "model": "${MODEL}",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpg;base64,${base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ]
        }
    ]
}
EOF

curl http://127.0.0.1:${PROXY_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @/tmp/request.json

rm -f /tmp/request.json

###############################################################################
# Benchmark
###############################################################################
echo "Running benchmark (stream)..."
vllm bench serve \
  --model               $MODEL \
  --backend             openai-chat \
  --endpoint            /v1/chat/completions \
  --dataset-name        random-mm \
  --seed                0 \
  --num-prompts         $NUM_PROMPTS \
  --port                $PROXY_PORT

PIDS+=($!)

# cleanup
echo "cleanup..."
cleanup