# c4ai-command-r-v01

## Introduction

CohereLabs/c4ai-command-r-v01 is an open-weights instruction model optimized for reasoning, summarization, and question answering. It supports multilingual generation and long-context processing, and is suitable for enterprise-style assistant and RAG scenarios. This tutorial describes how to deploy and validate the model on vLLM-Ascend with Atlas A2.

## Supported Features

Refer to the [supported models matrix](../../user_guide/support_matrix/supported_models.md) for the feature support status of this model.

## Environment Preparation

### Model Weight

`CohereLabs/c4ai-command-r-v01` (BF16): requires **4 × Atlas 800I A2 (64G)** NPUs. [Download model weight](https://www.modelscope.cn/models/AI-ModelScope/c4ai-command-r-v01)

It is recommended to download the model weight to a shared cache directory (for example, `/root/.cache/`).

## Installation

Run in a Docker container:

```{code-block} bash
:substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

## Deployment

This document is validated on vLLM-Ascend 0.17.0rc2.dev with CANN 8.5.1 on Atlas 800I A2.

```bash
vllm serve "CohereLabs/c4ai-command-r-v01" \
  --served-model-name c4ai-command-r-v01 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

## Functional Verification

Open another terminal and run:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "c4ai-command-r-v01",
    "messages": [
      {"role": "user", "content": "Hello, please reply with: test successful"}
    ],
    "max_tokens": 32,
    "temperature": 0.2
  }'
```

Expected result:

- HTTP status is `200 OK`.
- Response JSON contains `choices`.
- `finish_reason` is `stop`.

## Accuracy Evaluation

Reference accuracy values:

| Task | Metric | Expected (yaml) |
| --- | --- | --- |
| gsm8k | exact_match,strict-match | 0.445 |
| gsm8k | exact_match,flexible-extract | 0.569 |
