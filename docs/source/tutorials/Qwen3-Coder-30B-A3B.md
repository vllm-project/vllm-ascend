# Qwen3-Coder-30B-A3B

## Introduction

The newly released Qwen3-Coder-30B-A3B employs a sparse MoE architecture for efficient training and inference, delivering significant optimizations in agentic coding, extended context support of up to 1M tokens, and versatile function calling.

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen3-Coder-30B-A3B-Instruct`(BF16 version): require 1 Atlas 800 A3 (64G × 16) nodes or 1 Atlas 800 A2 (64G/32G × 8) nodes. [Download model weight](https://modelers.cn/models/Modelers_Park/Qwen3-Coder-30B-A3B-Instruct)

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Verify Multi-node Communication(Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../installation.md#verify-multi-node-communication).

### Installation

You can using our official docker image and install extra operator for supporting `Qwen3-Coder-30B-A3B-Instruct`.

:::{note}
Only AArch64 architecture are supported currently due to extra operator's installation limitations.
:::

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

1. Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

2. Install the package `custom-ops` to make the kernels available.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/CANN-custom_ops-sfa-linux.aarch64.run
chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
./CANN-custom_ops-sfa-linux.aarch64.run --quiet
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
```

::::
::::{tab-item} A2 series
:sync: A2

1. Start the docker image on your node, refer to [using docker](../installation.md#set-up-using-docker).

2. Install the package `custom-ops` to make the kernels available.

```shell
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/CANN-custom_ops-sfa-linux.aarch64.run
chmod +x ./CANN-custom_ops-sfa-linux.aarch64.run
./CANN-custom_ops-sfa-linux.aarch64.run --quiet
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/a2/custom_ops-1.0-cp311-cp311-linux_aarch64.whl
pip install custom_ops-1.0-cp311-cp311-linux_aarch64.whl
```

::::
:::::

In addition, if you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../installation.md).

- Install extra operator for supporting `Qwen3-Coder-30B-A3B-Instruct`, refer to the above tab.

If you want to deploy multi-node environment, you need to set up environment on each node.

## Deployment

### Single-node Deployment

Run the following script to execute online inference.

For an Atlas A2 with 64 GB of NPU card memory, tensor-parallel-size should be at least 2, and for 32 GB of memory, tensor-parallel-size should be at least 4.

```shell
#!/bin/sh
export VLLM_USE_MODELSCOPE=true

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --served-model-name qwen3-coder --tensor-parallel-size 4 --enable_expert_parallel
```

### Prefill-Decode Disaggregation

Not supported yet.

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
