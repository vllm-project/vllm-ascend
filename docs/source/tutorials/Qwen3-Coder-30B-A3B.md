# Qwen3-Coder-30B-A3B

## Introduction

The newly released Qwen3-Coder-30B-A3B employs a sparse MoE architecture for efficient training and inference, delivering significant optimizations in agentic coding, extended context support of up to 1M tokens, and versatile function calling.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

`Qwen3-Coder-30B-A3B-Instruct`(BF16 version): requires 1 Atlas 800 A3 node (with 16x 64G NPUs) or 1 Atlas 800 A2 node (with 8x 64G/32G NPUs). [Download model weight](https://modelers.cn/models/Modelers_Park/Qwen3-Coder-30B-A3B-Instruct)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

`Qwen3-Coder` is first supported in `vllm-ascend:v0.10.0rc1`, please run this model using a later version.

You can using our official docker image to run `Qwen3-Coder-30B-A3B-Instruct` directly.

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc1
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
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

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../installation.md).

## Deployment

### Single-node Deployment

Run the following script to execute online inference.

For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 2, and for 32 GB of memory, tensor-parallel-size should be at least 4.

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name qwen3-coder --tensor-parallel-size 4 --enable_expert_parallel
```

## Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3-coder",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 4096
}'
```

## Accuracy Evaluation

### Using AISBench

1. Refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result, here is the result of `Qwen3-Coder-30B-A3B-Instruct` in `vllm-ascend:0.11.0rc0` for reference only.

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| openai_humaneval | f4a973 | humaneval_pass@1 | gen | 94.51 |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.
