# Qwen2.5-7B

## Introduction

Qwen2.5-7B-Instruct is the flagship instruction-tuned variant of Alibaba Cloud’s Qwen 2.5 LLM series. It supports a maximum context window of 128K, enables generation of up to 8K tokens, and delivers enhanced capabilities in multilingual processing, instruction following, programming, mathematical computation, and structured data handling.

This document details the complete deployment and verification workflow for the model, including supported features, environment preparation, single-node deployment, functional verification, accuracy and performance evaluation, and troubleshooting of common issues. It is designed to help users quickly complete model deployment and validation.

The `Qwen2.5-7B-Instruct` model was supported since `vllm-ascend:v0.9.0`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen2.5-7B-Instruct`(BF16 version): require 1 Atlas 910B4 (32G × 1) card. [Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)

It is recommended to download the model weights to a local directory (e.g., `./Qwen2.5-7B-Instruct/`) for quick access during deployment.

### Installation

You can use our official docker image and install extra operator for supporting `Qwen2.5-7B-Instruct`.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

1. Start the docker image on your each node.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
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
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

Start the docker image on your each node.

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
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
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

### Single-node Deployment

Qwen2.5-7B-Instruct supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `./Qwen2.5-7B-Instruct/` directory.
2. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

vllm serve ${MODEL_PATH} \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name qwen-2.5-7b-instruct \
          --trust-remote-code \
          --max-model-len 32768
```

### Multi-node Deployment

Single-node deployment is recommended.

### Prefill-Decode Disaggregation

Not supported yet.

## Functional Verification

After starting the service, verify functionality using a `curl` request:

```shell
curl http://<IP>:<Port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen-2.5-7b-instruct",
        "prompt": "Beijing is a",
        "max_completion_tokens": 5,
        "temperature": 0
    }'
```

A valid response (e.g., `"Beijing is a vibrant and historic capital city"`) indicates successful deployment.

## Accuracy Evaluation

### Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

Results and logs are saved to `benchmark/outputs/default/`. A sample accuracy report is shown below:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- |--------------|
| gsm8k | - | accuracy | gen | 75.00  |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen2.5-7B-Instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve \
  --model ./Qwen2.5-7B-Instruct/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

After about several minutes, you can get the performance evaluation result.
