# Phi-4-multimodal-instruct

## Introduction

Phi-4-multimodal is a multimodal model from Microsoft, designed for vision and language understanding.

This document will show the main verification steps of the `Phi-4-multimodal-instruct`.

## Supported Features

- Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.
- Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Prepare Model Weights

Running this model requires 1 Atlas 800I A2 (64G × 8) node or 1 Atlas 800 A3 (64G × 16) node.

Download model weight at [Hugging Face](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) or download by below command:

```bash
pip install huggingface_hub
huggingface-cli download microsoft/Phi-4-multimodal-instruct --local-dir /root/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct
```

It is recommended to download the model weights to the shared directory of multiple nodes, such as `/root/.cache/`.

### Installation

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

Setup environment variables:

```bash
# Set `max_split_size_mb` to reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
```

## Deployment

### Single-node Deployment

Run the following command inside the container to start the vLLM server:

```{code-block} bash
   :substitutions:
vllm serve microsoft/Phi-4-multimodal-instruct \
--trust-remote-code \
--max-model-len 4096
```

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
    "model": "microsoft/Phi-4-multimodal-instruct",
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}},
            {"type": "text", "text": "What is in this image?"}
        ]}
    ],
    "max_completion_tokens": 100
    }'
```

## Accuracy Evaluation

Phi-4-multimodal on vllm-ascend can be tested using AISBench.

### Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

## Performance Evaluation

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Phi-4-multimodal-instruct` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve --model microsoft/Phi-4-multimodal-instruct --dataset-name random --random-input 1024 --num-prompt 200 --request-rate 1 --trust-remote-code --save-result --result-dir ./
```

After about several minutes, you can get the performance evaluation result.
