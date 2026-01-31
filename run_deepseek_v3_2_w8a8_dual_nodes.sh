#!/bin/bash
# Test script for DeepSeek-V3.2-W8A8 model on dual nodes (A3)

# Usage:
#   For master node (rank 0): ./run_deepseek_v3_2_w8a8_dual_nodes.sh master
#   For worker node (rank 1): ./run_deepseek_v3_2_w8a8_dual_nodes.sh worker <MASTER_IP>

NODE_TYPE=${1:-master}
MASTER_IP=${2:-"127.0.0.1"}
LOCAL_IP=${LOCAL_IP:-"127.0.0.1"}
SERVER_PORT=8080

echo "Running as: $NODE_TYPE"
echo "Master IP: $MASTER_IP"
echo "Local IP: $LOCAL_IP"

# Common environment variables
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_MODELSCOOP=true
export HCCL_BUFFSIZE=1024
export SERVER_PORT=8080
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_A3_EBA_ENABLE=1

if [ "$NODE_TYPE" = "master" ]; then
    # Master node (rank 0-1) - with HTTP server
    echo "Starting master node with HTTP server..."
    vllm serve vllm-ascend/DeepSeek-V3.2-W8A8 \
        --host 0.0.0.0 \
        --port $SERVER_PORT \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-address $LOCAL_IP \
        --data-parallel-rpc-port 13399 \
        --tensor-parallel-size 8 \
        --quantization ascend \
        --seed 1024 \
        --enable-expert-parallel \
        --max-num-seqs 16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 4096 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --compilation-config '{"cudagraph_capture_sizes": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
        --tokenizer-mode deepseek_v32 \
        --reasoning-parser deepseek_v3

elif [ "$NODE_TYPE" = "worker" ]; then
    # Worker node (rank 2-3) - headless
    echo "Starting worker node (headless)..."
    vllm serve vllm-ascend/DeepSeek-V3.2-W8A8 \
        --headless \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-rpc-port 13399 \
        --data-parallel-start-rank 2 \
        --data-parallel-address $MASTER_IP \
        --tensor-parallel-size 8 \
        --quantization ascend \
        --seed 1024 \
        --enable-expert-parallel \
        --max-num-seqs 16 \
        --max-model-len 8192 \
        --max-num-batched-tokens 4096 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --speculative-config '{"num_speculative_tokens": 2, "method":"deepseek_mtp"}' \
        --compilation-config '{"cudagraph_capture_sizes": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
        --tokenizer-mode deepseek_v32 \
        --reasoning-parser deepseek_v3

else
    echo "Invalid node type: $NODE_TYPE"
    echo "Usage:"
    echo "  Master node: $0 master"
    echo "  Worker node: $0 worker <MASTER_IP>"
    exit 1
fi
