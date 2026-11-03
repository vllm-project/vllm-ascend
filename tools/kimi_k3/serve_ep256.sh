#!/usr/bin/env bash
set -Eeuo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
unset HCCL_OP_EXPANSION_MODE
exec vllm serve "${KIMI_EP256_MODEL:?Set KIMI_EP256_MODEL}" \
  --served-model-name kimi-k3-ep256 \
  --host 127.0.0.1 --port "${KIMI_API_PORT:-2000}" \
  --trust-remote-code --tokenizer-mode kimi_k3 \
  --quantization ascend --generation-config vllm --reasoning-parser kimi_k3 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --dtype bfloat16 --enforce-eager \
  --max-model-len 6144 --max-num-seqs 100 \
  --max-num-batched-tokens 2048 --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"vision_chunk":0}' \
  --additional-config '{"enable_flashcomm1":true}' \
  --seed 42
