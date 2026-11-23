#!/usr/bin/env bash

export PYTHONPATH="$PWD/vllm:$PWD/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_V2_MODEL_RUNNER=1

exec vllm serve /home/hy4nxt \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --served-model-name hy4-flash --trust-remote-code \
  --max-num-seqs 128 --max-model-len 131072 --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 \
  --seed 1024 --enable-prefix-caching --async-scheduling \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,3,4,5,6,7,8,16,32,64,128],"max_decode_cudagraph_capture_tokens":512}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5}' \
  --additional-config '{"enable_mlapo":false,"enable_fused_mc2":0,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":false}}' \
  "$@"
