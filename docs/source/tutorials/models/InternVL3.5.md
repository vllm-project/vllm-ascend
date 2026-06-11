# InternVL3.5(InternVL3_5-38B/241B-A28B)

## Introduction

[InternVL3.5](https://huggingface.co/papers/2508.18265), a new family of open-source multimodal models that significantly advances versatility, reasoning capability, and inference efficiency along the InternVL series.

The `InternVL3.5` model is first supported in `vllm-ascend:v0.20.2`

This document will show the main verification steps of both `InternVL3_5-38B` and `InternVL3_5-241B-A28B` model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

require 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node:

- `InternVL3_5-38B`:   [Download model weight](https://huggingface.co/OpenGVLab/InternVL3_5-38B)
- `InternVL3_5-241B-A28B`:   [Download model weight](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)

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
--name $NAME \
--net=host \
--privileged=true \
--shm-size=500g \
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
-it $IMAGE bash
```

::::
::::{tab-item} Build from source
:sync: multi

You can build all from source.

- Install `vllm-ascend`, refer to [set up using python](../../installation.md#set-up-using-python).

::::
:::::

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Online Serving

:::::{tab-set}
:sync-group: install

::::{tab-item} InternVL3_5-38B
:sync: multi-node inference

Run docker container to start the vLLM server on multi-node NPU:

```{code-block} bash
   :substitutions:
vllm serve OpenGVLab/InternVL3_5-38B \
--tensor-parallel-size 2 \
--dtype bfloat16 \
--max_model_len 16384 \
--max-num-batched-tokens 16384
```

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "OpenGVLab/InternVL3_5-38B",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/blob/main/images/DvD.jpg"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-d3270d4a16cb4b98936f71ee3016451f","object":"chat.completion","created":1764924127,"model":"OpenGVLab/InternVL3_5-38B","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is: **DVD**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":107,"total_tokens":123,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO 12-05 08:42:07 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-38B
INFO 12-05 08:42:11 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:60988 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-05 08:42:13 [loggers.py:127] Engine 000: Avg prompt throughput: 10.7 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 12-05 08:42:23 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

::::
::::{tab-item} InternVL3_5-241B-A28B
:sync: multi

Run docker container to start the vLLM server on multi-NPU:

```shell
# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"
# Set vLLM to Engine V1
export VLLM_USE_V1=1

vllm serve OpenGVLab/InternVL3_5-241B-A28B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 16 \
    --max-model-len 30000 \
    --max-num-batched-tokens 50000 \
    --max-num-seqs 30 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --dtype bfloat16

```

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
    "model": "OpenGVLab/InternVL3_5-241B-A28B",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/blob/main/images/DvD.jpg"}},
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-c07088bf992a4b77a89d79480122a483","object":"chat.completion","created":1764905884,"model":"OpenGVLab/InternVL3_5-241B-A28B","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is:\n\n**OpenGVLab/InternVL3_5-241B-A28B**","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":73,"total_tokens":89,"completion_tokens":16,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Logs of the vllm server:

```bash
INFO 12-05 08:50:57 [chat_utils.py:560] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-241B-A28B
2025-12-05 08:50:58,913 - modelscope - INFO - Target directory already exists, skipping creation.
INFO 12-05 08:51:00 [acl_graph.py:187] Replaying aclgraph
INFO:     127.0.0.1:50720 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 12-05 08:51:10 [loggers.py:127] Engine 000: Avg prompt throughput: 7.3 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 12-05 08:51:20 [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

::::
:::::

## Accuracy Evaluation

### Using Language Model Evaluation Harness

The accuracy of some models is already within our CI monitoring scope, including:

As an example, take the `mmmu_val` dataset as a test dataset, and run accuracy evaluation of `InternVL3_5-241B-A28B` in offline mode.

1. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for more details on `lm_eval` installation.

    ```shell
    pip install lm_eval
    ```

2. Run `lm_eval` to execute the accuracy evaluation.

    ```shell
    lm_eval \
        --model vllm-vlm \
        --model_args pretrained=OpenGVLab/InternVL3_5-38B,max_model_len=8192,gpu_memory_utilization=0.7 \
        --tasks mmmu_val \
        --batch_size 32 \
        --apply_chat_template \
        --trust_remote_code \
        --output_path ./results
    ```

## Performance

### Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

The performance evaluation must be conducted in an online mode. Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model OpenGVLab/InternVL3_5-38B  --dataset-name random --random-input 200 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
