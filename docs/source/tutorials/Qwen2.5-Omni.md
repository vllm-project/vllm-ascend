# Qwen2.5-Omni-7B

## Introduction

Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.

The `Qwen2.5-Omni` model was supported since `vllm-ascend:v0.11.0rc0`. This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-NPU and multi-NPU deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen2.5-Omni-3B`(BF16): [Download model weight](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- `Qwen2.5-Omni-7B`(BF16): [Download model weight](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

Following examples use the 7B version deafultly.

### Installation

You can using our official docker image to run `Qwen2.5-Omni` directly.

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
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

## Deployment

### Single-node Deployment

#### Single NPU (Qwen2.5-Omni-7B)

```bash
export VLLM_USE_MODELSCOPE=true
export MODEL_PATH=vllm-ascend/Qwen2.5-Omni-7B
export LOCAL_MEDIA_PATH=/local_path/to_media/

vllm serve ${MODEL_PATH}\
--host 0.0.0.0 \
--port 8000 \
--served-model-name Qwen-Omni \
--allowed-local-media-path ${LOCAL_MEDIA_PATH} \
--trust-remote-code \
--compilation-config {"full_cuda_graph": 1} \
--no-enable-prefix-caching
```

:::{note}
Now vllm-ascend docker image should contain vllm[audio] build part, if you encounter *audio not supported issue* by any chance, please re-build vllm with [audio] flag.

```bash
VLLM_TARGET_DEVICE=empty pip install -v ".[audio]"
```

:::

`--allowed-local-media-path` is optional, only set it if you need infer model with local media file

`--gpu-memory-utilization` should not be set manually only if yous know what this parameter aims to.

#### Multiple NPU (Qwen2.5-Omni-7B)

```bash
export VLLM_USE_MODELSCOPE=true
export MODEL_PATH=vllm-ascend/Qwen2.5-Omni-7B
export LOCAL_MEDIA_PATH=/local_path/to_media/
export DP_SIZE=8

vllm serve ${MODEL_PATH}\
--host 0.0.0.0 \
--port 8000 \
--served-model-name Qwen-Omni \
--allowed-local-media-path ${LOCAL_MEDIA_PATH} \
--trust-remote-code \
--compilation-config {"full_cuda_graph": 1} \
--data-parallel-size ${DP_SIZE} \
--no-enable-prefix-caching
```

`--tensor_parallel_size` no need to set for this 7B model, but if you really need tensor parallel, tp size can be one of `1\2\4`

### Prefill-Decode Disaggregation

Not supported yet

## Functional Verification

If your service start successfully, you can see the info shown below:

```bash
INFO:     Started server process [2736]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Once your server is started, you can query the model with input prompts:

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer EMPTY"   -d '{
    "model": "Qwen-Omni",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is the text in the illustrate?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

```

If you query the server successfully, you can see the info shown below (client):

```bash
{"id":"chatcmpl-a70a719c12f7445c8204390a8d0d8c97","object":"chat.completion","created":1764056861,"model":"Qwen-Omni","choices":[{"index":0,"message":{"role":"assistant","content":"The text in the illustration is \"TONGYI Qwen\".","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":73,"total_tokens":88,"completion_tokens":15,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## Accuracy Evaluation

Qwen2.5-Omni on vllm-ascend has been test on AISBench.

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen2.5-Omni-7B` with `vllm-ascend:0.11.0rc0` for reference only.

| dataset | platform | metric | mode | vllm-api-stream-chat |
|----- | ----- | ----- | ----- | -----|
| textVQA | A2 | accuracy | gen_base64 | 83.47 |
| textVQA | A3 | accuracy | gen_base64 | 84.04 |

## Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen2.5-Omni-7B` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommand:
- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model vllm-ascend/Qwen2.5-Omni-7B --dataset-name random --random-input 1024 --num-prompt 200 --request-rate 1 --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
