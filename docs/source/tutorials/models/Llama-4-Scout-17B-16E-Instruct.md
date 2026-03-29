# Llama-4-Scout-17B-16E-Instruct

## Introduction

The **Llama-4-Scout-17B-16E-Instruct** is Meta's latest generation of Mixture-of-Experts (MoE) models, featuring a sophisticated **16-expert architecture**. It provides state-of-the-art reasoning and multilingual capabilities for complex inference tasks.

This document outlines the deployment and verification process on the **vLLM-Ascend** platform. To support Llama-4's unique MoE routing, kernel-level adaptations have been implemented to ensure stability and optimal performance on **Huawei Ascend Atlas A2** hardware.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Llama-4-Scout-17B-16E-Instruct` (BF16 version): requires at least 4 Atlas 800 A2 NPUs (TP=4).

It is recommended to download the model weight to the shared directory of the node, such as `/root/.cache/`.

### Installation

You can use our official docker image to run `Llama-4-Scout-17B-16E-Instruct` directly. Ensure you mount 4 NPU devices to satisfy the $2^n$ parallel constraint.

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

Start the docker image on your node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-llama4 \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/models \
    -it $IMAGE bash
```

::::

:::::

## Deployment

:::{note}

In this tutorial, we suppose you downloaded the model weight to `/models/llama4-scout`. Feel free to change it to your own path.

**Critical Kernel Patch:** This MoE model requires `attention_v1.py` to be configured with `sparse_mode=0` and a flattened `actual_seq_lengths_q` workaround. These changes resolve **ACL Error 507034** (stream synchronization failure) caused by Llama-4's TND layout on Ascend NPUs.

:::

### Single-node Deployment

Run the following script to execute online inference.

Shell

```
export HCCL_INTRA_ROCE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1

python3 -m vllm.entrypoints.openai.api_server \
    --model /models/llama4-scout \
    --served-model-name llama4-scout \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code \
    --block-size 128
```

## Functional Verification

Test the deployment using a standard OpenAI-compatible request:

Shell

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama4-scout",
    "messages": [{"role": "user", "content": "Write a Python script for quicksort."}],
    "temperature": 0
  }'
```

## Accuracy Evaluation

The reasoning capabilities of Llama-4-Scout have been verified using **EvalScope**.

| **Dataset** | **Samples** | **Metric** | **Score** |
| ----------- | ----------- | ---------- | --------- |
| **GSM8K**   | 100         | mean_acc   | **0.94**  |

### Reproduction Command

Shell

```
evalscope eval \
    --model llama4-scout \
    --api-url http://localhost:8000/v1 \
    --datasets gsm8k \
    --limit 100

