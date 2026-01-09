# GLM-4.5/4.6/4.7

## Introduction

GLM-4.x series models use a Mixture-of-Experts (MoE) architecture and are foundational models specifically designed for agent applications

The `GLM-4.5` model is first supported in `vllm-ascend:v0.10.0rc1`

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight
- `GLM-4.5`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-4.5).
- `GLM-4.6`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-4.6).
- `GLM-4.7`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-4.7).
- `GLM-4.5-w8a8-with-float-mtp`(Quantized version with mtp): [Download model weight](https://modelers.cn/models/Modelers_Park/GLM-4.5-w8a8).
- `GLM-4.6-w8a8`(Quantized version without mtp): [Download model weight](https://modelers.cn/models/Modelers_Park/GLM-4.6-w8a8). Because vllm do not support GLM4.6 mtp in October, so we do not provide mtp version. And last month, it supported, you can use the following quantization scheme to add mtp weights to Quantized weights.
- `Method of Quantify`: [quantization scheme](https://blog.csdn.net/qq_37368095/article/details/156429653?spm=1011.2124.3001.6209). You can use these methods to quantify the model.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

You can using our official docker image to run `GLM-4.x` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables
# Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
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

## Deployment

### Single-node Deployment

- In low-latency scenarios, we recommend a single-machine deployment.
- Quantized model `glm4.5_w8a8_with_float_mtp` can be deployed on 1 Atlas 800 A3 (64G × 16) or 1 Atlas 800 A2 (64G × 8).

Run the following script to execute online inference.

```shell
#!/bin/sh
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV

vllm serve /weight/glm4.5_w8a8_with_float_mtp \
  --data-parallel-size 1 \
  --tensor-parallel-size 16 \
  --seed 1024 \
  --served-model-name glm \
  --max-model-len 35000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"num_speculative_tokens": 1, "model":"/weight/glm4.5_w8a8_with_float_mtp", "method":"mtp"}' \
  --compilation-config '{"cudagraph_capture_sizes": [1,2,4,8,16,32], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --async-scheduling \
```

**Notice:**
The parameters are explained as follows:
- For single-node deployment, we recommend using `dp1tp16` and turn off expert parallel in low-latency scenarios.
- `--async-scheduling` Asynchronous scheduling is a technique used to optimize inference efficiency. It allows non-blocking task scheduling to improve concurrency and throughput, especially when processing large-scale models.

### Multi-node Deployment

Not recommended to deploy multi-node on Atlas 800 A2 (64G * 8).

### Prefill-Decode Disaggregation

Not test yet.

## Accuracy Evaluation

Here are two accuracy evaluation methods.

### Using AISBench
1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `GLM4.6` in `vllm-ascend:main` (after `vllm-ascend:0.13.0rc1`) for reference only.

| dataset | version | metric | mode | vllm-api-general-chat | note |
|----- | ----- | ----- | ----- | -----| ----- |
| gsm8k | - | accuracy | gen | 96.13 | 1 Atlas 800 A3 (64G × 16) |
| gsm8k | - | accuracy | gen | 96.06 | GPU |

### Using Language Model Evaluation Harness

Not test yet.

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `GLM-4.x` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve \
  --backend vllm \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 22400 \
  --prefix-repetition-suffix-len 9600 \
  --prefix-repetition-output-len 1024 \
  --num-prompts 1 \
  --prefix-repetition-num-prefixes 1 \
  --ignore-eos \
  --model glm \
  --tokenizer /weight/glm4.5_w8a8_with_float_mtp \
  --seed 1000 \
  --host 0.0.0.0 \
  --port 8000 \
  --endpoint /v1/completions \
  --max-concurrency 1 \
  --request-rate 1 \
```

After about several minutes, you can get the performance evaluation result.
