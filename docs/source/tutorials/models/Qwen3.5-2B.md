# Qwen3.5-2B

## Introduction

Qwen3.5 is the latest generation of Qwen series models, featuring hybrid Mamba + full attention architecture for efficient long-context reasoning. Qwen3.5-2B is the smallest dense model in the Qwen3.5 family, suitable for resource-constrained deployment and quick prototyping.

Qwen3.5-2B supports both text-only and multimodal (image) inputs, and features a built-in thinking mode for chain-of-thought reasoning.

This document will show the main verification steps of the model, including supported features, environment preparation, deployment, accuracy and performance evaluation.

The `Qwen3.5-2B` model is first supported in `vllm-ascend:v0.17.0rc1`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Model Information

| Item | Value |
|------|-------|
| Architecture | Qwen3_5ForConditionalGeneration (Hybrid Attention + Mamba) |
| Hidden Size | 2048 |
| Num Layers | 24 (6 full attention + 18 linear attention) |
| Attention Heads | 16 |
| KV Heads | 4 |
| Head Dim | 128 |
| Intermediate Size | 11008 |
| Vocab Size | 151936 |
| Precision | BF16 |
| Model Size | ~4.3 GB |
| Multimodal | Yes (text + image) |

## Environment Preparation

### Model Weight

- `Qwen3.5-2B` (BF16 version): require 1 Atlas 800I A2 (64G x 1) card. [Download from HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-2B).

It is recommended to download the model weight to a shared directory, such as `/root/.cache/`.

### Installation

:::::{tab-set}
::::{tab-item} Use docker image

Select an image based on your machine type and start the container. Refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache/:/root/.cache/ \
-it $IMAGE bash
```

::::
::::{tab-item} Build from source

- Install `vllm-ascend`, refer to [set up using python](../../installation.md#set-up-using-python).

::::
:::::

## Deployment

### Single-node Deployment (TP1)

Qwen3.5-2B requires only 1 NPU card (~4.3 GB).

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export OMP_NUM_THREADS=8

vllm serve Qwen/Qwen3.5-2B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 1 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 512 \
--gpu-memory-utilization 0.85 \
--trust-remote-code \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'
```

### Single-node Deployment (TP2)

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export OMP_NUM_THREADS=8
export HCCL_OP_EXPANSION_MODE=AIV
export MASTER_PORT=29500

vllm serve Qwen/Qwen3.5-2B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 2 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 256 \
--gpu-memory-utilization 0.85 \
--trust-remote-code \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'
```

### Single-node Deployment (TP4)

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export OMP_NUM_THREADS=8
export HCCL_OP_EXPANSION_MODE=AIV
export MASTER_PORT=29500

vllm serve Qwen/Qwen3.5-2B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 4 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 256 \
--gpu-memory-utilization 0.85 \
--trust-remote-code \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'
```

**Notice:**

- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` enables AclGraph for the decode phase, which is the primary performance optimization.
- `--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'` enables async exponential scheduling and CPU core binding, which provides significant throughput improvement.
- `--default-chat-template-kwargs '{"enable_thinking":false}'` can be added to disable the thinking mode for faster non-reasoning tasks.
- `HCCL_OP_EXPANSION_MODE=AIV` is recommended for TP > 1 to improve communication efficiency.

## Functional Verification

### Text Generation

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.5-2B",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

### Chat Completion

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.5-2B",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0
    }'
```

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, here are the results of `Qwen3.5-2B` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | metric | accuracy |
|---------|--------|----------|
| gsm8k | accuracy | 78.7% |
| ARC-Easy | accuracy | 88.0% |
| ARC-Challenge | accuracy | 81.7% |
| PIQA | accuracy | 75.8% |

## Performance

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

Standard benchmark parameters: `--dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 64 --seed 42 --burstiness 1.0`.

### Baseline Performance (enforce-eager)

| Config | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|--------|-------------|-------|----------------|----------------|
| TP1 | 1,393 | 10.88 | 1,256 | 36.3 |
| TP2 | 1,346 | 10.52 | 850 | 41.0 |
| TP4 | 1,060 | 8.28 | 1,444 | 46.9 |

### Optimized Performance (AclGraph + additional-config)

| Config | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) | Improvement |
|--------|-------------|-------|----------------|----------------|-------------|
| TP1 | 2,524 | 19.72 | 1,308 | 15.2 | +81.2% |
| TP2 | 3,320 | 25.94 | 822 | 12.8 | +146.6% |
| **TP4** | **4,117** | **32.16** | **716** | **9.9** | **+288.4%** |

**Key Findings:**

1. **TP4 is the optimal configuration**, achieving 4,117 tok/s.
2. **Positive TP scaling with optimization**: TP1(2,524) < TP2(3,320) < TP4(4,117).
3. **AclGraph + additional-config is the key optimization**: AclGraph alone provides ~+87% for TP2/TP4, and `enable_async_exponential` + `enable_cpu_binding` provides the additional boost.
4. **Baseline shows negative TP scaling** due to HCCL communication overhead, but optimization reverses this.
