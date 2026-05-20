# CohereLabs/c4ai-command-r-v01

## Introduction

CohereLabs/c4ai-command-r-v01 is an open-weights instruction model optimized for reasoning, summarization, and question answering. It supports multilingual generation and long-context processing, and is suitable for enterprise-style assistant and RAG scenarios. This tutorial describes how to deploy and validate the model on vLLM-Ascend with Atlas A2.

## Supported Features

Refer to the [supported models matrix](../../user_guide/support_matrix/supported_models.md) for the feature support status of this model.

## Environment Preparation

### Model Weight

`CohereLabs/c4ai-command-r-v01` (BF16): requires **4 × Atlas 800I A2 (64G)** NPUs. [Model weight](https://huggingface.co/CohereLabs/c4ai-command-r-v01)

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

```{test} bash
:sync-yaml: tests/e2e/models/configs/c4ai-command-r-v01.yaml
:sync-target: test_cases[0].model test_cases[0].server_cmd
:sync-class: cmd

vllm serve "CohereLabs/c4ai-command-r-v01" \
  --served-model-name c4ai-command-r-v01 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
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

Run the LM-Eval correctness test with the model config:

```bash
python -m pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/c4ai-command-r-v01.yaml \
  --tp-size 4 \
  --report-dir ./benchmarks/accuracy
```

Reference values (from `tests/e2e/models/configs/c4ai-command-r-v01.yaml`):

| Task | Metric | Expected (yaml) |
| --- | --- | --- |
| gsm8k | exact_match,strict-match | 0.20 |
| gsm8k | exact_match,flexible-extract | 0.15 |

## FAQ

### HCCL init failure (EI0010 / error code 5)

**Symptoms:**

- `hcclCommInitRootInfoConfig ... error code is 5`
- `P2P_Communication_Failed(EI0010)`

**Recommended checks:**

- Confirm visible NPU count matches `--tensor-parallel-size` (this model uses TP4).
- Verify multi-card interconnect and driver/toolkit installation in your deployment environment.
- Retry after ensuring all workers use consistent device mapping.
