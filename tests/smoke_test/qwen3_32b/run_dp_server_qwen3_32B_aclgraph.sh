#!/bin/bash
set -euo pipefail

weight_dir="slanyer/Qwen3_32B_INT8"

export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
#export ASCEND_RT_VISIBLE_DEVICES=0,1
# AIV
export HCCL_OP_EXPANSION_MODE="AIV"
# MASK
#export PAGED_ATTENTION_MASK_LEN= 5500
#图模式--aclgraph
vllm serve "$weight_dir" \
  --host 0.0.0.0 \
  --port 20002 \
  --no-enable-prefix-caching \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3 \
  --max-model-len 36784 \
  --max-num-batched-tokens 36784 \
  --block-size 128 \
  --trust-remote-code \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --additional-config '{"enable_weight_nz_layout":true}' 
