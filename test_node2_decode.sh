#!/bin/bash

# Node 2 - Decode (kv_consumer) - Master
# 16 NPU, DP size 8 (local 4, ranks 0-3), TP 4
# Requires: LOCAL_IP environment variable

export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_MODELSCOPE=true
export SERVER_PORT=8080
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE=1024
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
export VLLM_TORCH_PROFILER_WITH_STACK=0
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export VLLM_ASCEND_ENABLE_MLAPO=1
export TASK_QUEUE_ENABLE=1

# Auto-detect LOCAL_IP if not set
# You can override by setting: export LOCAL_IP="<your_ip>"
: ${LOCAL_IP:=$(hostname -I | awk '{print $2}')}
echo "Using LOCAL_IP: $LOCAL_IP"

vllm serve vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $LOCAL_IP \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --quantization ascend \
    --max-model-len 68000 \
    --max-num-batched-tokens 12 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4, 8, 12, 16]}' \
    --trust-remote-code \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --additional-config '{"enable_cpu_binding" : false,"recompute_scheduler_enable" : true}' \
    --tokenizer-mode deepseek_v32 \
    --reasoning-parser deepseek_v3 \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {
                "dp_size": 2,
                "tp_size": 16
            },
            "decode": {
                "dp_size": 8,
                "tp_size": 4
            }
        }
    }'
