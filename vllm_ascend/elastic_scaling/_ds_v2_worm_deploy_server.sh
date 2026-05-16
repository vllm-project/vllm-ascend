#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$(BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

cd $SCRIPT_DIR

export HEAD_IP=$(hostname -I | awk '{print $1}')

export VLLM_USE_V1=1
export VLLM_VERSION=0.14.1
export MODEL_PATH=/path/to/your/model
export WORLD_SIZE=16
export MODEL_SIZE=4
export TP_SIZE=4
export NPU_START_IDX=0
export PORT_NUM=8000

export BLOCK_SIZE=16
export KV_GPU_BLOCKS=100
export SKIP_LOAD_FROM_DISK=""

export HCCL_HOST_SOCKET_PORT_RANGE="60100-60150"
export HCCL_NPU_SOCKET_PORT_RANGE="60100-60150"

python worm_server/worm_server.py