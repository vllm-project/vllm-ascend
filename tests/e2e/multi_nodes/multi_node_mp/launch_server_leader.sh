#!/bin/bash

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
local_ip=$(hostname -I | awk '{print $1}')
nic_name=eth0


export VLLM_USE_MODELSCOPE=True
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export HCCL_BUFFSIZE=1024

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
bash $SCRIPT_DIR/installer.sh


vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 8080 \
--data-parallel-size 2 \
--data-parallel-size-local 1 \
--data-parallel-address $local_ip \
--data-parallel-rpc-port 13389 \
--tensor-parallel-size 16 \
--seed 1024 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 32768 \
--quantization ascend \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--gpu-memory-utilization 0.9 \
--additional-config '{"torchair_graph_config":{"enabled":true}}'
