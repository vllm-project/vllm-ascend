# Qwen3-VL-30B-A3B-Instruct

## Introduction

The Qwen-VL(Vision-Language) series from Alibaba Cloud comprises a family of powerful Large Vision-Language Models (LVLMs) designed for comprehensive multimodal understanding. They accept images, text, and bounding boxes as input, and output text and detection boxes, enabling advanced functions like image detection, multi-modal dialogue, and multi-image reasoning.

This document will show the main verification steps of the `Qwen3-VL-30B-A3B-Instruct`, including supported features, feature configuration, environment preparation, NPU deployment.

## Supported Features

- Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.
- Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

require 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node:

- `Qwen3-VL-30B-A3B-Instruct`: [Download model weight](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct).

A sample Qwen3-VL-MoE quantization script can be found in the modelslim code repository: [Qwen3-VL-MoE Quantization Script Example](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/multimodal_vlm/Qwen3-VL-MoE/README.md).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

:::::{tab-set}
:sync-group: install

::::{tab-item} single-NPU
:sync: single

Run docker container:

```{code-block} bash
   :substitutions:
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

::::
::::{tab-item} multi-NPU
:sync: multi

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
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
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /data:/data \
-it $IMAGE bash
```

::::
:::::

Setup environment variables:

```bash
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

:::{note}
`max_split_size_mb` prevents the native allocator from splitting blocks larger than this size (in MB). This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory. You can find more details [<u>here</u>](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/envref/envref_07_0061.html).
:::

## Deployment

### Offline Inference

:::::{tab-set}
:sync-group: install

::::{tab-item} Image Inputs
:sync: multi

Run the following script to execute offline inference on multi-NPU:

```bash
pip install qwen_vl_utils
```

```python

```

If you run this script successfully, you can see the info shown below:

```

```

::::
::::{tab-item} Video Inputs
:sync: multi

Run the following script to execute offline inference on multi-NPU:

```bash
pip install qwen_vl_utils
```

```python

```

If you run this script successfully, you can see the info shown below:

```

```

::::
:::::

### Online Serving

:::::{tab-set}
:sync-group: install

::::{tab-item} Image Inputs
:sync: multi

Run docker container to start the vLLM server on multi-NPU:

```{code-block} bash
   :substitutions:
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
--max_model_len 16384 \
--max-num-batched-tokens 16384 \
--tensor-parallel-size 2 \
--enable-expert-parallel
```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-30B-A3B-Instruct model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [746077]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
            {"type": "text", "text": "What is the text in the illustrate?"}
        ]}
    ],
    "max_tokens": 100
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-974cb7a7a746a13e","object":"chat.completion","created":1766569357,"model":"/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is \"TONGYI Qwen\".","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":122,"completion_tokens":15,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO 12-24 09:42:37 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:54946 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-24 09:42:41 [loggers.py:257] Engine 000: Avg prompt throughput: 10.7 tokens/s, Avg generation throughput: 1.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
```

::::
::::{tab-item} Video Inputs
:sync: multi

Run docker container to start the vLLM server on multi-NPU:

```shell
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
--max_model_len 16384 \
--max-num-batched-tokens 16384 \
--tensor-parallel-size 2 \
--enable-expert-parallel \
--allowed-local-media-path /media
```

:::{note}
Add `--max_model_len` option to avoid ValueError that the Qwen3-VL-30B-A3B-Instruct model's max seq len (262144) is larger than the maximum number of tokens that can be stored in KV cache. This will differ with different NPU series base on the HBM size. Please modify the value according to a suitable value for your NPU series.
:::

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [746077]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "file:///media/test.mp4"}},
            {"type": "text", "text": "What is in this video?"}
        ]}
    ],
    "max_tokens": 100
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-a03c6d6e40267738","object":"chat.completion","created":1766569752,"model":"/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The video shows a standard test pattern, which is a series of vertical bars in various colors (red, green, blue, yellow, magenta, cyan, and white) arranged in a circular pattern on a black background. This is a common visual used in television broadcasting to calibrate and test equipment. The pattern remains static throughout the video.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":196,"total_tokens":266,"completion_tokens":70,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO:     127.0.0.1:49314 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-24 09:49:22 [loggers.py:257] Engine 000: Avg prompt throughput: 19.6 tokens/s, Avg generation throughput: 7.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, MM cache hit rate: 33.3%
```

::::
:::::
