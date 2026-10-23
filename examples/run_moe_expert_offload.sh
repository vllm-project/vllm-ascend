#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

model_path=${1:?Usage: run_moe_expert_offload.sh LOCAL_CHECKPOINT_PATH [extra vllm serve arguments]}
shift

# Run in an environment exposing one idle NPU. Extra CLI arguments can override
# these acceptance defaults, including --kv-cache-dtype fp8 and the slot count.
export VLLM_USE_V2_MODEL_RUNNER=1
exec vllm serve "$model_path" \
    --served-model-name moe-offload --host 127.0.0.1 --port 18001 \
    --tensor-parallel-size 1 --block-size 32 --kv-cache-dtype bfloat16 \
    --no-async-scheduling --max-model-len 2048 \
    --max-num-batched-tokens 512 --max-num-seqs 4 --enable-chunked-prefill \
    --kv-cache-memory-bytes 2147483648 --seed 0 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
    --additional-config '{"expert_offload_config":{"enabled":true,"num_device_experts":16}}' \
    "$@"
