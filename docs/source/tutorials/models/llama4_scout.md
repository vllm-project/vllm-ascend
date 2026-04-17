# Llama-4-Scout-17B-16E-Instruct

## Introduction

Llama-4-Scout-17B-16E-Instruct is a high-performance Mixture-of-Experts (MoE) large language model developed by Meta. It features a 16-expert architecture that optimizes both reasoning capabilities and computational efficiency. By activating only a subset of experts per token, it delivers state-of-the-art performance in complex logical reasoning and multimodal understanding.

This document introduces the deployment of the Llama-4-Scout model on Huawei Ascend NPU hardware using the vLLM-Ascend platform.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Key features supported for Llama-4-Scout:

* **Precision**: Bfloat16 (BF16).
* **Parallelism**: Tensor Parallel (TP) size 1, 2, 4, 8.
* **Engine**: vLLM V1 Engine with optimized NPU attention kernels.

## Environment Preparation

### Model Weight

* **Llama-4-Scout-17B-16E-Instruct**: Requires at least 2 Atlas 800 A2 (64G) NPUs to accommodate weights and inference overhead. [Download from HuggingFace](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)

### Installation

You can use the official docker image to run the model directly. Refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
:substitutions:
# Update --device according to your resource (Minimum 2 NPUs required for weight loading).
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

docker run --rm \
--name $NAME \
--shm-size=16g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /data/models:/data/models \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

## Deployment

### Service-oriented Deployment

For Llama-4-Scout, we recommend using Tensor Parallel (TP) size 2 as the minimum configuration on Atlas A2. For optimal performance in production, TP=4 is recommended.

```shell
#!/bin/sh

export MODEL_PATH=/data/models/Llama-4-Scout-17B-16E-Instruct

vllm serve ${MODEL_PATH} \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --enforce-eager \
  --trust-remote-code \
  --gpu-memory-utilization 0.90
```

**Notice:**

* `--tensor-parallel-size 2` is used here to satisfy the $2^n$ minimum card requirement for the 17B model weights.
* If you encounter OOM (Out of Memory) with long sequences, consider increasing the card count to TP=4.

## Accuracy Evaluation

The accuracy of Llama-4-Scout has been verified using the Language Model Evaluation Harness (lm_eval) on the GSM8K dataset.

| Dataset | Model | Hardware | TP Size | Metric | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gsm8k | Llama-4-Scout-17B | Atlas A2 | 4 | accuracy | 0.96 |

### Verification Command

```shell
lm_eval \
  --model local-completions \
  --model_args model=${MODEL_PATH},base_url=http://localhost:8000/v1/completions \
  --tasks gsm8k \
  --output_path ./
```

## Performance

By implementing optimized list-based sequence length passing for NPU kernels, this adaptation minimizes Host-to-Device synchronization overhead.

* **Latency**: Demonstrated significantly lower prefill latency compared to the baseline MoE implementation.
* **Throughput**: High efficiency achieved through optimized MoE routing and expert-parallel scaling.
