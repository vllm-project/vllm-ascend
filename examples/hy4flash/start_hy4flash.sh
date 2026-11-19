#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="$script_dir/vllm:$script_dir/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"

# The host lib64 mount can hide the OpenSSL libraries bundled in the image.
openssl_dir="$script_dir/openssl"
if [[ -d "$openssl_dir" ]]; then
  export LD_LIBRARY_PATH="$openssl_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_V2_MODEL_RUNNER=1

max_num_seqs="${HY4FLASH_MAX_NUM_SEQS:-128}"
case "${HY4FLASH_GRAPH_MODE:-decode}" in
  eager) mode_args=(--enforce-eager) ;;
  decode)
    # Use vLLM's 512-token decode capture budget. Keep small batches exact
    # and let intermediate batches reuse the next larger FULL graph.
    capture_requests="$(python3 - "$max_num_seqs" <<'PY'
import json
import sys

limit = min(int(sys.argv[1]), 512)
sizes = list(range(1, min(limit, 8) + 1))
size = 16
while size < limit:
    sizes.append(size)
    size *= 2
if limit > 8:
    sizes.append(limit)
print(json.dumps(sizes))
PY
)"
    mode_args=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"decode_cudagraph_capture_request_sizes\":$capture_requests,\"max_decode_cudagraph_capture_tokens\":512}")
    ;;
  *) printf '%s\n' 'HY4FLASH_GRAPH_MODE must be eager or decode' >&2; exit 2 ;;
esac

case "${HY4FLASH_ENABLE_DFLY:-1}" in
  0) ;;
  1) mode_args+=(--speculative-config '{"method":"dfly","num_speculative_tokens":5}') ;;
  *) printf '%s\n' 'HY4FLASH_ENABLE_DFLY must be 0 or 1' >&2; exit 2 ;;
esac

case "${HY4FLASH_SHARED_EXPERT_MULTISTREAM:-1}" in
  0) shared_expert_multistream=false ;;
  1) shared_expert_multistream=true ;;
  *) printf '%s\n' 'HY4FLASH_SHARED_EXPERT_MULTISTREAM must be 0 or 1' >&2; exit 2 ;;
esac

additional_config="{\"enable_mlapo\":false,\"enable_fused_mc2\":0,\"multistream_overlap_shared_expert\":${shared_expert_multistream},\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}"

exec vllm serve "${HY4FLASH_CHECKPOINT:-/home/hy4nxt}" \
  --host "${HY4FLASH_HOST:-127.0.0.1}" --port "${HY4FLASH_PORT:-8000}" \
  --tensor-parallel-size "${HY4FLASH_TP:-8}" \
  --served-model-name hy4-flash \
  --max-num-seqs "$max_num_seqs" \
  --max-model-len "${HY4FLASH_MAX_MODEL_LEN:-131072}" \
  --max-num-batched-tokens "${HY4FLASH_MAX_NUM_BATCHED_TOKENS:-4096}" \
  --enable-expert-parallel \
  --trust-remote-code \
  --kv-cache-dtype "${HY4FLASH_KV_CACHE_DTYPE:-fp8}" \
  --gpu-memory-utilization "${HY4FLASH_GPU_MEMORY_UTILIZATION:-0.85}" \
  --seed 1024 \
  "${mode_args[@]}" \
  --enable-prefix-caching \
  --async-scheduling \
  --additional-config "$additional_config" \
  "$@"
