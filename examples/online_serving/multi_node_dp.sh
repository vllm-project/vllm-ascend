#!/bin/bash

set -euo pipefail

run_node() {
    NODE_TYPE=$1
    echo "====> Running $NODE_TYPE"

    local_ip=$(hostname -I | awk '{print $1}')
    iface=$(ip -o -4 addr show | awk -v ip="$local_ip" '$4 ~ ip"/" {print $2}')

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$iface
    export TP_SOCKET_IFNAME=$iface
    export HCCL_SOCKET_IFNAME=$iface
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=100
    export VLLM_USE_V1=1
    export HCCL_BUFFSIZE=1024

    if [ "$NODE_TYPE" == "header" ]; then
        echo "====> Running header node"
        vllm serve /root/.cache/weights/Kimi-K2-Instruct-W8A8 \
        --host 0.0.0.0 \
        --port 8004 \
        --data-parallel-size 4 \
        --api-server-count 2 \
        --data-parallel-size-local 2 \
        --data-parallel-address $local_ip \
        --data-parallel-rpc-port 13389 \
        --seed 1024 \
        --served-model-name kimi \
        --quantization ascend \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --max-num-seqs 16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 4096 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 \
        --additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
    else
        echo "====> Running worker node"
        vllm serve /root/.cache/weights/Kimi-K2-Instruct-W8A8 \
        --host 0.0.0.0 \
        --port 8004 \
        --headless \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-start-rank 2 \
        --data-parallel-address $MASTER_ADDR \
        --data-parallel-rpc-port 13389 \
        --seed 1024 \
        --tensor-parallel-size 8 \
        --served-model-name kimi \
        --max-num-seqs 16 \
        --max-model-len 32768 \
        --quantization ascend \
        --max-num-batched-tokens 4096 \
        --enable-expert-parallel \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.92 \
        --additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":true}}'
    fi
}

run_node "$@"