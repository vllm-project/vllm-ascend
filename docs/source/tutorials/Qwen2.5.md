# Qwen2.5-Instruct

## Introduction

Qwen2.5-Instruct is the flagship instruction-tuned variant of Alibaba Cloud’s Qwen 2.5 LLM series. It supports a maximum context window of 128K, enables generation of up to 8K tokens, and delivers enhanced capabilities in multilingual processing, instruction following, programming, mathematical computation, and structured data handling.

This document details the complete deployment and verification workflow for the model, including supported features, environment preparation, single-node deployment, functional verification, accuracy and performance evaluation, and troubleshooting of common issues. It is designed to help users quickly complete model deployment and validation.

The `Qwen2.5-Instruct` model was supported since `vllm-ascend:v0.11.0rc0`.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen2.5-7B-Instruct`(BF16 version): require 2 910B4 (32G × 2) cards. [Qwen2.5-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)

It is recommended to download the model weights to a local directory (e.g., `./Qwen2.5-Instruct/`) for quick access during deployment.

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image and install extra operator for supporting `Qwen2.5-Instruct`.

:::{note}
Only AArch64 architecture are supported currently due to extra operator's installation limitations.
:::

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
    --device /dev/davinci1 \
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
    --device /dev/davinci1 \
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

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../installation.md).

- Install extra operator for supporting `Qwen2.5-7B-Instruct`, refer to the above tab.

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

Qwen2.5-Instruct supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `./Qwen2.5-7B-Instruct/` directory.
2. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PAGED_ATTENTION_MASK_LEN=max_seq_len
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export ASCEBD_RT_VISIBLE_DEVICES=0,1

vllm serve ./Qwen2.5-Instruct/ \
          --host <IP> \
          --port <Port> \
          --served-model-name qwen-2.5-7b-instruct \
          --trust-remote-code \
          --max-model-len 32768 \
          --tensor-parallel-size 2 \
          --enforce-eager
```

### Multi-node Deployment

This document currently focuses on single-node deployment.

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
        "max_tokens": 5,
        "temperature": 0
    }'
```

A valid response (e.g., `"Beijing is a vibrant and historic capital city"`) indicates successful deployment.

## Accuracy Evaluation

Two accuracy evaluation methods are provided: AISBench (recommended) and manual testing with standard datasets.

### Using AISBench

Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

Results and logs are saved to `benchmark/outputs/default/`. A sample accuracy report is shown below:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- |--------------|
| gsm8k | - | accuracy | gen | 75.00  |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark
Run performance evaluation of `Qwen2.5-Instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=true
vllm bench serve \
  --model ./Qwen2.5-Instruct/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompt 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

After about several minutes, you can get the performance evaluation result.
