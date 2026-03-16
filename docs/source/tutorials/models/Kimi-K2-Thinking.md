# Kimi-K2-Thinking

## Introduction

Kimi-K2-Thinking is a large-scale Mixture-of-Experts (MoE) model developed by Moonshot AI. It features a hybrid thinking architecture that excels in complex reasoning and problem-solving tasks.

This document will show the main verification steps of the model, including supported features, environment preparation, single-node deployment, and functional verification.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Kimi-K2-Thinking`(bfloat16): require 1 Atlas 800 A3 (64G × 16) node. [Download model weight](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

It is recommended to download the model weight to the shared directory, such as `/mnt/sfs_turbo/.cache/`.

### Installation

You can use our official docker image to run `Kimi-K2-Thinking` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

## Run with Docker

```{code-block} bash
   :substitutions:
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
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /mnt/sfs_turbo/.cache:/home/cache \
-it $IMAGE bash
```

## Verify the Quantized Model

Please be advised to edit the value of `"quantization_config.config_groups.group_0.targets"` from `["Linear"]` into `["MoE"]` in `config.json` of original model downloaded from [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

```json
{
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": [
          "MoE"
        ]
      }
    }
  }
}
```

Your model files look like:

```bash
.
|-- chat_template.jinja
|-- config.json
|-- configuration_deepseek.py
|-- configuration.json
|-- generation_config.json
|-- model-00001-of-000062.safetensors
|-- ...
|-- model-00062-of-000062.safetensors
|-- model.safetensors.index.json
|-- modeling_deepseek.py
|-- tiktoken.model
|-- tokenization_kimi.py
`-- tokenizer_config.json
```

## Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

For an Atlas 800 A3 (64G*16) node, tensor-parallel-size should be at least 16.

```bash
vllm serve Kimi-K2-Thinking \
--served-model-name kimi-k2-thinking \
--tensor-parallel-size 16 \
--enable-expert-parallel \
--trust-remote-code \
--no-enable-prefix-caching
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "kimi-k2-thinking",
  "messages": [
    {"role": "user", "content": "Who are you?"}
  ],
  "temperature": 1.0
}'
```
