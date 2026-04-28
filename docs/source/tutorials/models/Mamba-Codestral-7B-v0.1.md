# Mamba-Codestral-7B-v0.1

## Introduction

Mamba-Codestral-7B-v0.1 (Codestral Mamba) is an open-source code generation model based on the Mamba2 architecture, developed by Mistral AI. It performs on par with state-of-the-art Transformer-based code models while offering linear-time inference scaling with respect to sequence length.

The model architecture is `Mamba2ForCausalLM`. It has been verified to run on the `vllm-ascend` stack using the existing Mamba support path.

This document covers environment preparation and single-node deployment of `mistralai/Mamba-Codestral-7B-v0.1` on Atlas A2 series hardware.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

## Environment Preparation

### Model Weight

Download the model weights from Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mamba-Codestral-7B-v0.1",
    local_dir="/root/.cache/Mamba-Codestral-7B-v0.1",
)
```

### Installation

Use the official vllm-ascend Docker image. Select an image based on your machine type and start the container:

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 series
:sync: A2

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
:::::

## Deployment

### Single-card Deployment

`Mamba-Codestral-7B-v0.1` can be deployed on a single Atlas 800 A2 card.

Run the following command to start the vllm serving endpoint:

```shell
vllm serve /root/.cache/Mamba-Codestral-7B-v0.1 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

**Notes:**

- `--trust-remote-code` is required to resolve the model architecture correctly.
- `--no-enable-prefix-caching` is recommended for the first verification pass. Mamba cache mode and prefix caching interaction is sensitive on Ascend; enable it only after the baseline is confirmed.
- Adjust `--max-model-len` based on your available NPU memory.

### Sending a Request

After the server starts, send a test generation request:

```shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/Mamba-Codestral-7B-v0.1",
    "prompt": "def fibonacci(n):",
    "max_tokens": 128
  }'
```

## Accuracy Evaluation

Run the GSM8K benchmark using the vllm-ascend e2e test framework:

```shell
python -m pytest tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/Mamba-Codestral-7B-v0.1.yaml \
  --tp-size 1 \
  --report-dir reports \
  -s
```

Expected results on Atlas A2 Series (1 card):

| Task  | Metric                       | Value  |
|-------|------------------------------|--------|
| gsm8k | exact_match, strict-match    | 0.1243 |
| gsm8k | exact_match, flexible-extract| 0.1259 |
