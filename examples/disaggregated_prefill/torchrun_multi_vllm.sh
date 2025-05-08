#!/bin/bash
export HCCL_IF_IP=2.0.0.0
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=0

export ASCEND_RT_VISIBLE_DEVICES=0,1
# export VLLM_DP_SIZE=2
# export VLLM_DP_RANK=0
export VLLM_DP_MASTER_IP="2.0.0.0"
export VLLM_DP_MASTER_PORT=40001
export VLLM_DP_PROXY_IP="2.0.0.0"
export VLLM_DP_PROXY_PORT=30002
export VLLM_DP_MONITOR_PORT=30003
# export VLLM_HTTP_PORT=20001

MASTER_ADDR=4.0.0.0
MASTER_PORT=12345

# number of nodes
NNODES=2
# number of vllms in each node
NPROC_PER_NODE=8
# rank of current node
NODE_RANK=0

# 启动分布式任务
torchrun \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  run_single_vllm.py