#!/bin/bash

set -eo errexit

. $(dirname "$0")/common.sh

export VLLM_ENABLE_MC2=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_VERSION=0.9.1
# FIXME: unset HCCL_OP_EXPANSION_MODE to avoid the torch_air bug
unset HCCL_OP_EXPANSION_MODE

MODEL_NAME="vllm-ascend/DeepSeek-V3-Pruning"
TP_SIZE=4
DP_SIZE=4
REGISTER_PORT=10102
ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NUM_REDUNDANT_EXPERT=$1


function run_eplb_instance() {
    local model_name=$1
    local tp_size=$2
    local dp_size=$3
    local register_port=$4
    local num_redundant_expert=$5

    _info "====> Test model: $model_name"
    _info "====> TP size: $tp_size"
    _info "====> DP size: $dp_size"
    _info "====> Register port: $register_port"
    _info "====> Expert map path: ./tests/e2e/eplb/expert_map.json"
    _info "====> Num redundant expert: $num_redundant_expert"

    ASCEND_RT_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES vllm serve $model_name \
    --host 0.0.0.0 \
    --port $register_port \
    --tensor-parallel-size $tp_size \
    --data-parallel-size $dp_size \
    --enable-expert-parallel \
    --served-model-name Deepseek \
    --max-model-len 8192 \
    --max-num-seqs 24 \
    --trust-remote-code \
    --additional-config '{"torchair_graph_config": {"enabled": true, "graph_batch_sizes": [24]}, "ascend_scheduler_config": {"enabled": true}, "expert_map_path": "./tests/e2e/eplb/expert_map.json"}'
}


_info "====> Start staic_eplb test"
run_eplb_instance $MODEL_NAME $TP_SIZE $DP_SIZE $REGISTER_PORT $NUM_REDUNDANT_EXPERT