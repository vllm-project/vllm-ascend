# Qwen3-Next

## 1 Introduction

The Qwen3-Next model is a sparse MoE (Mixture of Experts) model with high sparsity. Compared to the MoE architecture of Qwen3, it has introduced key improvements in aspects such as the hybrid attention mechanism and multi-token prediction mechanism, enhancing the training and inference efficiency of the model under long contexts and large total parameter scales.

This document will present the core verification steps of the model, including supported features, environment preparation, as well as accuracy and performance evaluation. Qwen3 Next is currently using Triton Ascend, which is in the experimental phase. In subsequent versions, its performance related to stability and accuracy may change, and performance will be continuously optimized.

The `Qwen3-Next` model is first supported in `vllm-ascend:v0.10.2rc1` and can stably run in v0.16.0 and later version.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

`Qwen3-Next-80B-A3B-Instruct`: requires **8 cards in 1 Atlas 800 A3 (64G × 16) node** or **8 cards in 1 Atlas 800 A2 (64G × 8) node**. [Model Weight](https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct)

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

**A3 series:**

Start the docker image on your each node.

```bash
#!/bin/sh
# Update the vllm-ascend image
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
--shm-size=1g \
--name vllm-ascend-qwen3 \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
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

The Qwen3 Next is using [Triton Ascend](https://gitee.com/ascend/triton-ascend) which is currently experimental. In future versions, there may be behavioral changes related to stability, accuracy, and performance improvement.

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

For more details, please refer to the [Installation Guide](../../installation.md).

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source. Install vLLM from source first:

1. Clone and install vLLM:

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

2. Clone and install the vLLM-Ascend repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm vllm-ascend
```

Expected result: The version information for both packages is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

For more details, please refer to the [Installation Guide](../../installation.md).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. The model `Qwen3-Next-80B-A3B-Instruct` can be deployed on 1 Atlas 800 A3 (64G × 16).

While a single-node setup supports all input/output scenarios, consider deploying multinodes for optimal performance.

Startup Command:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --served-model-name qwen3_next --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3_next",
  "messages": [
    {"role": "user", "content": "Who are you?"}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_completion_tokens": 32
}'
```

Expected Result:

The service returns HTTP 200 OK with a JSON response containing the `choices` field. Example output (content truncated for brevity):

```json
{
    "id": "chatcmpl-9df13fd5e539af93",
    "object": "chat.completion",
    "created": 1780971952,
    "model": "qwen3_next",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "What do you know about me?\n\nHello! I am Qwen, a large-scale language model independently developed by the Tongyi Lab under Alibaba Group. I am...",
                "reasoning": "The user is asking for my thoughts on \"Who are you?\"...",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "token_ids": null
        }
    ]
}
```

## 7 Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3-Next-80B-A3B-Instruct` in `vllm-ascend:0.13.0rc1` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| gsm8k | - | accuracy | gen | 95.53 |

## 8 Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen3-Next` as an example.

Refer to [vLLM Benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model Qwen/Qwen3-Next-80B-A3B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

The performance result is:  

```bash
Hardware: A3-752T, 2 node
Deployment: TP4 + Full Decode Only
Input/Output: 2k/2k
Concurrency: 32
Performance: 580tps, TPOT 54ms
```

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**:
>
> - The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.
>
> - Qwen3-Next does not support TP>=16 now. Since this model has 16 query heads but only 2 key and value heads, GQA degenerates into MHA when TP >= 16. However, the FIA operator currently fails to function in MHA scenarios with a head dimension of 256 (which is the case for this model).

#### Table 1: Scenario Overview

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|High Throughput<br>(16K context)|Single-Node Mixed|2 (A3)|Qwen3-Next|Use tp2 for high-resolution text inputs|
|Long Context<br>(128K, no prefix cache)|Single-Node Mixed|2 (A3)|Qwen3-Next|tp2 for high-resolution text inputs|
|Long Context<br>(128K, with prefix cache)|Single-Node Mixed|2 (A3)|Qwen3-Next|tp2 for high-resolution text inputs|
|Multimodal<br>(1080P)|Single-Node Mixed|2 (A3)|Qwen3-Next|tp2 for high-resolution visual inputs|

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64G × 16 NPUs).

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|-------------------|--------------------|
|High Throughput / Low Latency (16K)|Server / Single Machine|2|1|1|~16K|3|
|Long Context (128K, no cache)|Server / Single Machine|2|1|1|128K|3|
|Long Context (128K, with cache)|Server / Single Machine|2|1|1|128K|3|
|Multimodal (1080P)|Server / Single Machine|2|1|1|~16K|3|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html).
