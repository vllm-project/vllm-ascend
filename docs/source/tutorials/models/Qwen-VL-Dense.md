# Qwen-VL-Dense(Qwen3-VL-2B/4B/8B/32B)

## 1 Introduction

The Qwen-VL(Vision-Language)series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, NPU deployment, accuracy and performance evaluation.

This tutorial uses the vLLM-Ascend `v0.11.0rc3-a3` version for demonstration, showcasing the `Qwen3-VL-8B-Instruct` model as an example for single NPU and multi-NPU deployment.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

require 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node:

- `Qwen3-VL-2B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)
- `Qwen3-VL-4B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct)
- `Qwen3-VL-8B-Instruct`:   [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)
- `Qwen3-VL-32B-Instruct`:  [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

**A3 series**

Start the docker image on your each node.

``` bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

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
-p 8000:8000 \
-it $IMAGE bash
```

After a successful docker run, you can verify the running container service by executing the `docker ps` command.

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy multi-node environment, you need to set up environment on each node.

To use the tools_call feature, please ensure that your transformers version is 4.57.6 or lower. If vllm-ascend has been upgraded to v0.21 or later, this requirement no longer applies.

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Run docker container to start the vLLM server on single-NPU:

``` bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
--dtype bfloat16 \
--max_model_len 16384 \
--max-num-batched-tokens 16384
```

Key Parameter Descriptions:

- Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-8B-Instruct model's max seq len (256000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series based on the on-chip memory size. Please modify the value according to a suitable value for your NPU series.

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Service Verification:

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

Expected Result:
If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-d3270d4a16cb4b98936f71ee3016451f","object":"chat.completion","created":1764924127,"model":"Qwen/Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is: **TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":123,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```
### 5.2 Multi-Node PD Separation Deployment

Run docker container to start the vLLM server on multi-NPU:

```shell
#!/bin/sh
# if os is Ubuntu
apt update
apt install libjemalloc2 
# if os is openEuler
yum update
yum install jemalloc
# Add the LD_PRELOAD environment variable
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    # On Ubuntu, first install with `apt install libjemalloc2`
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
elif [ -f /usr/lib64/libjemalloc.so.2 ]; then
    # On openEuler, first install with `yum install jemalloc`
    export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
fi
# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"
# Set vLLM to Engine V1
export VLLM_USE_V1=1

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 30000 \
    --max-num-batched-tokens 50000 \
    --max-num-seqs 30 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --dtype bfloat16

```

Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-32B-Instruct model's max_model_len (128000) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the on-chip memory size. Please modify the value according to a suitable value for your NPU series.

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [14431]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-c07088bf992a4b77a89d79480122a483","object":"chat.completion","created":1764905884,"model":"Qwen/Qwen3-VL-32B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is:\n\n**TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":73,"total_tokens":89,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
INFO 12-05 08:50:57 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-32B-Instruct
2025-12-05 08:50:58,913 - modelscope - INFO - Target directory already exists, skipping creation.
INFO 12-05 08:51:00 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:50720 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-05 08:51:10 [loggers.py:127] Engine 000: Avg prompt throughput: 7.3 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 12-05 08:51:20 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

Expected Result:

The service returns HTTP 200 OK. 
```bash
{"id":"chatcmpl-d3270d4a16cb4b98936f71ee3016451f","object":"chat.completion","created":1764924127,"model":"Qwen/Qwen3-VL-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is: **TONGYI Qwen**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":123,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## 7 Accuracy Evaluation

Here is one accuracy evaluation method.

### Using Language Model Evaluation Harness

The accuracy of some models is already within our CI monitoring scope, including:

- `Qwen3-VL-8B-Instruct`

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `Qwen3-VL-8B-Instruct` in offline mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

    ```shell
    pip install lm_eval
    ```

2. Run `lm_eval` to execute the accuracy evaluation.

    ```shell
    lm_eval \
        --model vllm-vlm \
        --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_model_len=8192,gpu_memory_utilization=0.7 \
        --tasks mmmu_val \
        --batch_size 32 \
        --apply_chat_template \
        --trust_remote_code \
        --output_path ./results
    ```

3. After execution, you can get the result, here is the result of `Qwen3-VL-8B-Instruct` in `vllm-ascend:0.11.0rc3` for reference only.

    |  Tasks  |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
    |---------|------:|------|-----:|------|---|-----:|---|-----:|
    |mmmu_val |      0|none  |      |acc   |↑  |0.5389|±  |0.0159|

## 8 Performance Evaluation

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

The performance evaluation must be conducted in an online mode. Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model Qwen/Qwen3-VL-8B-Instruct  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|High Throughput<br>(16K context)|Single-Node Mixed|2 (A3)|Qwen3-VL-32B-Instruct|Use tp2 for high-resolution text inputs|
|Long Context<br>(128K, no prefix cache)|Single-Node Mixed|2 (A3)|Qwen3-VL-32B-Instruct|tp2 for high-resolution text inputs|
|Long Context<br>(128K, with prefix cache)|Single-Node Mixed|2 (A3)|Qwen3-VL-32B-Instruct|tp2 for high-resolution text inputs|
|Multimodal<br>(1080P)|Single-Node Mixed|2 (A3)|Qwen/Qwen3-VL-32B-Instruct|tp2 for high-resolution visual inputs|

#### Table 2: Detailed Node Configuration

|Scenario|Configuration|NPUs|TP|DP|Max Model Len|MTP Speculation Num|
|--------|-------------|-----|--|--|-------------------|--------------------|
|High Throughput / Low Latency (16K)|Server / Single Machine|2|2|1|~16K|3|
|Long Context (128K, no cache)|Server / Single Machine|2|2|1|128K|3|
|Long Context (128K, with cache)|Server / Single Machine|2|2|1|128K|3|
|Multimodal (1080P)|Server / Single Machine|2|2|1|~16K|3|

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Chapter 5](#5-online-service-deployment).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-online-service-deployment)** chapter.

## 10 FAQ