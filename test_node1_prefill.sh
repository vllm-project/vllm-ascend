#!/bin/bash

# Node 1 - Prefill (kv_producer)
# 16 NPU, DP rank 1, TP 16

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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve vllm-ascend/DeepSeek-V3.2-W8A8 \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --data-parallel-size 2 \
    --data-parallel-start-rank 1 \
    --data-parallel-size-local 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --seed 1024 \
    --quantization ascend \
    --max-num-seqs 64 \
    --max-model-len 68000 \
    --max-num-batched-tokens 32550 \
    --trust-remote-code \
    --gpu-memory-utilization 0.82 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --additional-config '{"enable_cpu_binding" : false, "enable_sfa_cp":false,"layer_sharding": ["q_b_proj", "o_proj"]}' \
    --tokenizer-mode deepseek_v32 \
    --reasoning-parser deepseek_v3 \
    --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
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
