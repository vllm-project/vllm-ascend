# Llama-4-Scout-17B-16E-Instruct on vLLM-Ascend

## Introduction

The **Llama-4-Scout-17B-16E-Instruct** is Meta's latest generation of Mixture-of-Experts (MoE) models, featuring a sophisticated **16-expert architecture**. It provides state-of-the-art reasoning and multilingual capabilities for complex inference tasks.

This document outlines the deployment and verification process on the **vLLM-Ascend** platform. To support Llama-4's unique MoE routing, kernel-level adaptations have been implemented to ensure stability and optimal performance on **Huawei Ascend Atlas A2** hardware.

## Supported Features

| Feature | Status | Configuration |
| :--- | :--- | :--- |
| **BF16 Inference** | Supported | `--dtype bfloat16` |
| **Tensor Parallel** | Supported | `--tensor-parallel-size 4` |
| **MoE Support** | Supported | 16-Expert Routing |
| **Eager Mode** | Required | `--enforce-eager` |

## Environment Preparation

### Environment Variables

Configure the following variables to ensure HCCL communication stability and proper operator binding. Replace `/path/to/...` with your actual directory if different:

```bash
# Enable Intra-ROCE for HCCL stability
export HCCL_INTRA_ROCE_ENABLE=1

# NPU Library Paths
export NPU_LIB_DIR=/usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib
export LIBRARY_PATH=$LIBRARY_PATH:$NPU_LIB_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NPU_LIB_DIR:/vllm-workspace/vllm-ascend/vllm_ascend

# vLLM Python Path
export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
```

## Deployment

### Single-node Deployment (Atlas A2)

Llama-4-Scout-17B-16E requires 4 NPUs (TP4) for stable inference with a 1024 context length.

```bash
#!/bin/bash
# Save as start_llama4.sh
python3 -m vllm.entrypoints.openai.api_server \
    --model /data/models/llama4-scout \
    --served-model-name llama4-scout \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code \
    --block-size 128
```

> **Note:**
> **Critical Kernel Patch:** This model requires `attention_v1.py` to be configured with `sparse_mode=0` and a flattened `actual_seq_lengths_q` workaround. These changes resolve **ACL Error 507034** (stream synchronization failure) caused by Llama-4's TND layout on Ascend NPUs.

## Functional Verification

### Chat Completion API

Test the deployment using a standard OpenAI-compatible request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama4-scout",
    "messages": [{"role": "user", "content": "Write a Python script for quicksort."}],
    "temperature": 0
  }'
```

## Accuracy Evaluation (GSM8K)

The reasoning capabilities of Llama-4-Scout have been verified using **EvalScope**.

| Dataset | Samples | Metric | Score |
| :--- | :--- | :--- | :--- |
| **GSM8K** | 100 | mean_acc | **0.94** |

### Reproduction Command

```bash
evalscope eval \
    --model llama4-scout \
    --api-url http://localhost:8000/v1 \
    --datasets gsm8k \
    --limit 100
```
