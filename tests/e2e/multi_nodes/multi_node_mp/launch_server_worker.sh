#!/bin/bash

local_ip=$(hostname -I | awk '{print $1}')
nic_name=eth0

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
bash $SCRIPT_DIR/installer.sh
leader_ip=$(getent hosts $LWS_LEADER_ADDRESS | awk '{print $1}')

export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024

vllm serve $MODEL_PATH \
--headless \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-start-rank 1 \
--data-parallel-address $leader_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--quantization ascend \
--max-num-seqs 16 \
--max-model-len 32768 \
--max-num-batched-tokens 4096 \
--enable-expert-parallel \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--additional-config '{"torchair_graph_config":{"enabled":true}}'
