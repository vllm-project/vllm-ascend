#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.3.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
set -u
export VLLM_SERVER_DEV_MODE=1
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.link_priority":"PEER2PEER"}'
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export NO_PROXY='*'
export no_proxy='*'
export HCCL_IF_BASE_PORT=47500

export PYTHONPATH="/workspace/hy4-cann93/hy4-draft-only-20261126/vllm-src:/workspace/hy4-cann93/hy4-draft-only-20261126/source${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_HOST_IP=127.0.0.1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1

exec /usr/local/python3.12.13/bin/python "$SCRIPT_DIR/bin/vllm" serve /home/d00943518/weights/Hy4-flash-FP8 \
  --host 127.0.0.1 \
  --port 8100 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enforce-eager \
  --seed 1024 \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"engine_id":"prefix-dfly-p-1795082110","kv_connector_extra_config":{"ascend_local_comm_res_path":"/workspace/hy4-cann93/hy4-draft-only-20261126/pd-compat-06/communication-resources","prefill":{"tp_size":4,"dp_size":1},"decode":{"tp_size":1,"dp_size":4}}}' \
  --additional-config '{"enable_mlapo":false,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_flashcomm1":false}'
