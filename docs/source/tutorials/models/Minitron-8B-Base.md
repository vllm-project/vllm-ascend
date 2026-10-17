# Minitron-8B-Base

## 1 Introduction

The released `Minitron-8B-Base` is a lightweight, efficient large language model developed by NVIDIA. It is designed for general-purpose text generation and reasoning tasks, and can be deployed with vLLM for online serving and evaluation on Ascend NPU hardware through `vllm-ascend`.

This document describes the main verification steps of the model, including supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation on the GSM8K benchmark.

## 2 Environment Preparation

### 2.1 Model Weight

`Minitron-8B-Base`(BF16 version): requires 1 Ascend 910B (with 1 x 64GB NPUs). [Download model weight](https://www.modelscope.cn/models/nv-community/Minitron-8B-Base)

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or a local model path like `/data/vllm-workspace/models/Minitron-8B-Base`.

## 3 Installation

`Minitron-8B-Base` can be deployed with `vllm-ascend` in a compatible runtime environment.

### 3.1 Docker Image Installation

You can use the official docker image for deployment:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
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
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

### 3.2 Source Code Installation

If you do not want to use the docker image, you can also build from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## 4 Deployment

Start the online serving service with the following command:

``` bash
vllm serve "nv-community/Minitron-8B-Base" \
  --served-model-name minitron-8b-base \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

## 5 Functional Verification

Once your server is started, you can query the model with a simple prompt:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minitron-8b-base",
    "prompt": "Question: If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?\nAnswer:",
    "max_tokens": 64,
    "temperature": 1.0
  }'
```

A valid response indicates that the model is deployed correctly and can generate text outputs.

## 6 Accuracy Evaluation

The GSM8K dataset was used to evaluate the reasoning capability of `Minitron-8B-Base`.

The current evaluation setting is:

- Dataset: `gsm8k`
- Split: `test`
- Number of samples: `1000`
- Few-shot setting: `5-shot`
- `apply_chat_template`: `False`
- `fewshot_as_multiturn`: `False`

The current evaluation results are:

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Accuracy | gsm8k / test | Total Samples | 1000 |
| Accuracy | gsm8k / test | exact_match,strict-match | 0.5436 |
| Accuracy | gsm8k / test | exact_match,flexible-extract | 0.5451 |

### Remarks on Metrics

- **exact_match,strict-match**: Only predictions that strictly match the expected final-answer extraction format are counted as correct.
- **exact_match,flexible-extract**: Predictions are evaluated with a more flexible answer extraction rule, which tolerates minor formatting differences as long as the final numeric answer is correct.

## 7 Performance Evaluation

### 7.1 Baseline Result

`Minitron-8B-Base` can be deployed through `vllm-ascend` for online inference and benchmark evaluation.
Actual throughput and latency depend on hardware resources, prompt length, output length, concurrency, and runtime configuration.

### 7.2 Remarks

This document focuses on functional verification and benchmark accuracy on GSM8K.
Further benchmarking is recommended for:

- request latency
- throughput under concurrency
- long-context inference
- memory utilization
- stability under continuous serving workloads
