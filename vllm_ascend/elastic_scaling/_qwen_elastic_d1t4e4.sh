#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$(BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

cd $SCRIPT_DIR

## experimental settings
BLOCK_SIZE=16
MAX_NUM_SEQS=4
MAX_MODEL_LEN=1000
TP=4
DP=1

export HCCL_HOST_SOCKET_PORT_RANGE="60300-60400"
export HCCL_NPU_SOCKET_PORT_RANGE="60300-60400"
port=7101

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export VLLM_USE_V1=1
export VLLM_VERSION=0.14.1

export MODEL_PATH=/path/to/your/model
export ELASTIC=1
export KV_GPU_BLOCKS=100
export INFER_STATUS=0

export EXPERT_PARTITION_SPLIT=$TP
export LOCAL_NUM_EXPERTS=32
export GLOBAL_NUM_EXPERTS=128

python -m vllm_ascend.elastic_scaling.inference.api_server \
--model $MODEL_PATH \
--host 0.0.0.0 \
--port $port \
--data-parallel-size $DP \
--tensor-parallel-size $TP \
--seed 1024 \
--dtype float16 \
--served-model-name inference \
--enable-expert-parallel \
--max-num-seqs $MAX_NUM_SEQS \
--num-gpu-blocks-override $KV_GPU_BLOCKS \
--block-size $BLOCK_SIZE \
--max-model-len $MAX_MODEL_LEN \
--gpu-memory-utilization 0.95 \
--trust-remote-code \
--no-enable-prefix-caching \
--enforce-eager \

