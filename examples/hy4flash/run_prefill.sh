#!/usr/bin/env bash
set -euo pipefail

# Set these paths for the serving environment.
MODEL_PATH="<HY4_FLASH_FP8_PATH>"
VLLM_PATH="<VLLM_SOURCE_DIRECTORY>"
VLLM_ASCEND_PATH="<VLLM_ASCEND_SOURCE_DIRECTORY>"
UB_ENDPOINTS_PATH="<UB_ENDPOINT_CONFIG_DIRECTORY>"
SERVICE_PORT=8100
KV_PORT=30000
MAX_NUM_SEQS=128
MAX_MODEL_LEN=131072
MAX_BATCHED_TOKENS=4096
GPU_MEMORY_UTILIZATION=0.85

export PYTHONPATH="${VLLM_PATH}:${VLLM_ASCEND_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_HOST_IP=127.0.0.1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1

exec vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port "$SERVICE_PORT" \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --async-scheduling \
  --enforce-eager \
  --seed 1024 \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
  --kv-transfer-config '{
    "kv_connector":"MooncakeConnectorV2",
    "kv_role":"kv_producer",
    "kv_port":'"$KV_PORT"',
    "engine_id":"hy4-flash-prefill",
    "kv_connector_extra_config":{
      "ascend_local_comm_res_path":"'"$UB_ENDPOINTS_PATH"'",
      "prefill":{"tp_size":4,"dp_size":1},
      "decode":{"tp_size":1,"dp_size":4}
    }
  }' \
  --additional-config '{
    "multistream_overlap_shared_expert":true,
    "ascend_compilation_config":{"enable_npugraph_ex":true}
  }'
