#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 DEVICE_IDS PORT DP_SIZE DP_RANK DP_ADDRESS DP_RPC_PORT TP_SIZE PCP_SIZE" >&2
  exit 2
fi

export PYTHONPATH="/home/s00971618/hy4-flash-0day/Env/vllm:/home/s00971618/hy4-flash-0day/Env/vllm-ascend-private-codex-hy4-flash${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_HOST_IP=127.0.0.1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1

exec vllm serve /home/s00971618/weight/Hy4-flash-FP8-testing \
  --host 127.0.0.1 --port "$2" \
  --tensor-parallel-size "$7" --prefill-context-parallel-size "$8" \
  --data-parallel-size "$3" --data-parallel-size-local 1 \
  --data-parallel-external-lb --data-parallel-rank "$4" \
  --data-parallel-address "$5" --api-server-count 1 --device-ids "$1" \
  --data-parallel-rpc-port "$6" \
  --served-model-name hy4-flash \
  --dtype bfloat16 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --seed 1024 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,3,4,5,6,7,8,16,32,64,128],"max_decode_cudagraph_capture_tokens":512}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"engine_id":"prefix-dfly-d-1795082110","kv_connector_extra_config":{"ascend_local_comm_res_path":"/home/s00971618/hy4-flash-0day/UB8_1P1D/ub_endpoints","prefill":{"tp_size":4,"dp_size":1},"decode":{"tp_size":1,"dp_size":4}}}' \
  --additional-config '{"enable_mlapo":false,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":false},"hy4flash_dfly_dp_graph":true}'
