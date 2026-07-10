#!/bin/bash

export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

/usr/local/python3.11.14/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 3840 \
    --max-num-batched-tokens 3840 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000