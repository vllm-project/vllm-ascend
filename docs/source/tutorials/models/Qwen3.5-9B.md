# Qwen3.5-9B

## Introduction

Qwen3.5 is the latest generation of Qwen series models, featuring hybrid Mamba + full attention architecture for efficient long-context reasoning. Qwen3.5-9B is the largest dense model in the Qwen3.5 family that can run on a single Atlas 800I A2 card, offering strong reasoning capability within a compact resource footprint.

Qwen3.5-9B supports both text-only and multimodal (image) inputs, and features a built-in thinking mode for chain-of-thought reasoning.

This document will show the main verification steps of the model, including supported features, environment preparation, deployment, accuracy and performance evaluation.

The `Qwen3.5-9B` model is first supported in `vllm-ascend:v0.17.0rc1`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Model Information

| Item | Value |
|------|-------|
| Architecture | Qwen3_5ForConditionalGeneration (Hybrid Attention + Mamba) |
| Hidden Size | 3584 |
| Num Layers | 32 (8 full attention + 24 linear attention) |
| Attention Heads | 24 |
| KV Heads | 4 |
| Head Dim | 256 |
| Intermediate Size | 18944 |
| Vocab Size | 151936 |
| Precision | BF16 |
| Model Size | ~17.5 GB |
| Multimodal | Yes (text + image) |

## Environment Preparation

### Model Weight

- `Qwen3.5-9B` (BF16 version): require 1 Atlas 800I A2 (64G x 1) card. [Download from HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-9B).

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

Qwen3.5-9B requires 1 NPU card (~17.5 GB).

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export OMP_NUM_THREADS=8

vllm serve Qwen/Qwen3.5-9B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 1 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 256 \
--gpu-memory-utilization 0.85 \
--trust-remote-code \
--compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}' \
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

vllm serve Qwen/Qwen3.5-9B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 2 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 256 \
--gpu-memory-utilization 0.85 \
--trust-remote-code \
--compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}' \
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

vllm serve Qwen/Qwen3.5-9B \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 4 \
--max-model-len 4096 \
--max-num-batched-tokens 16384 \
--max-num-seqs 256 \
--gpu-memory-utilization 0.5 \
--trust-remote-code \
--compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'
```

**Notice:**

- `--additional-config '{"enable_async_exponential":true,"enable_cpu_binding":true}'` is the most critical optimization, providing approximately **+205%** throughput improvement for TP1.
- For TP4, `--gpu-memory-utilization 0.5` is recommended as AclGraph graph capture consumes additional memory on this model size.
- `--default-chat-template-kwargs '{"enable_thinking":false}'` can be added to disable the thinking mode for faster non-reasoning tasks.
- `HCCL_OP_EXPANSION_MODE=AIV` is recommended for TP > 1.

## Functional Verification

### Text Generation

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.5-9B",
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
        "model": "Qwen/Qwen3.5-9B",
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

2. After execution, here are the results of `Qwen3.5-9B` in `vllm-ascend:v0.17.0rc1` for reference only.

| dataset | metric | accuracy |
|---------|--------|----------|
| gsm8k | accuracy | 94.8% |

## Performance

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

Standard benchmark parameters: `--dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 64 --seed 42 --burstiness 1.0`.

### Baseline Performance (enforce-eager)

| Config | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|--------|-------------|-------|----------------|----------------|
| TP1 | 334 | 2.61 | 4,038 | 122.0 |
| TP2 | 334 | 2.61 | 2,534 | 132.0 |
| TP4 | 414 | 3.23 | 2,137 | 110.6 |

### Optimized Performance (AclGraph + additional-config)

| Config | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) | Improvement |
|--------|-------------|-------|----------------|----------------|-------------|
| TP1 | 1,019 | 7.96 | 3,345 | 36.8 | +205% |
| TP2 | 1,528 | 11.94 | 1,839 | 27.5 | +357% |
| **TP4** | **2,203** | **17.21** | **1,220** | **19.3** | **+432%** |

**Key Findings:**

1. **TP4 is the optimal configuration**, achieving 2,203 tok/s.
2. **Positive TP scaling with optimization**: TP1(1,019) < TP2(1,528) < TP4(2,203).
3. **`enable_async_exponential` + `enable_cpu_binding` is the key optimization**, providing the majority of throughput improvement.
4. **Baseline shows near-zero TP scaling** (TP1=TP2=334), but optimization restores positive scaling.
5. **Note**: AclGraph graph capture with `gpu-memory-utilization > 0.5` may cause OOM on TP4 for this model size. Use `--gpu-memory-utilization 0.5` or `--enforce-eager` if OOM occurs.
