# LLaVA-OneVision-Qwen2-0.5B-OV

## Introduction

`llava-hf/llava-onevision-qwen2-0.5b-ov-hf` is a compact multimodal model built on top of Qwen2. It supports text-only generation together with image understanding, multi-image reasoning, and visual dialogue.

This document shows the main verification steps for the model on vLLM Ascend, including environment preparation, single-NPU deployment, functional verification, and the existing accuracy baseline used by the repository.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`: [Download model weight](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)

The verified single-card deployment uses one Atlas A2 NPU. It is recommended to cache model weights under `/root/.cache` in advance to reduce startup time.

### Installation

You can use the official docker image to run `LLaVA-OneVision-Qwen2-0.5B-OV` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
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
    -it $IMAGE bash
```

## Deployment

### Single-node Deployment

#### Single NPU

Run the following script to start the vLLM service on a single Atlas A2 NPU:

```bash
export MODEL_PATH="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name LLaVA-OneVision-0.5B \
    --trust-remote-code \
    --gpu-memory-utilization 0.8
```

#### Multiple NPU

Single-NPU deployment is recommended for this 0.5B model.

### Prefill-Decode Disaggregation

Not supported yet.

## Functional Verification

If your service starts successfully, you can see logs similar to the following:

```bash
INFO:     Started server process [8173]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

You can first verify that the model is exposed by the OpenAI-compatible API:

```bash
curl http://127.0.0.1:8000/v1/models
```

### Text-only Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "LLaVA-OneVision-0.5B",
        "messages": [
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ],
        "max_completion_tokens": 16,
        "temperature": 0
    }'
```

If the request succeeds, you can see a response similar to the following:

```bash
{"choices":[{"message":{"content":"Hello! How can I assist you today?"}}]}
```

### Image Understanding Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "LLaVA-OneVision-0.5B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image briefly."
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
        "max_completion_tokens": 64,
        "temperature": 0
    }'
```

If the request succeeds, you can see a response similar to the following:

```bash
{"choices":[{"message":{"content":"The image features a logo consisting of a stylized geometric figure and the text \"TONGYI\" and \"Qwen\"..."}}]}
```

## Accuracy Evaluation

The repository already contains an end-to-end accuracy baseline for this model in `tests/e2e/models/configs/llava-onevision-qwen2-0.5b-ov-hf.yaml`.

| dataset | platform | metric | value |
|----- | ----- | ----- | ----- |
| ceval-valid | A2 | acc,none | 0.42 |
