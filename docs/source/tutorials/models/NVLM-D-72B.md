# NVLM-D-72B

## Introduction

NVLM-D-72B is a multimodal large language model (MLLM) developed by Nvidia. The model adopts a hybrid architecture of InternViT-6B visual encoder and Qwen2-72B-Instruct language model, supporting dynamic high-resolution image understanding with 1-D tile-tagging design.

## Environment Preparation

### Model Weight

- `NVLM-D-72B` (BF16 version): [Download model weight](https://www.modelscope.cn/models/AI-ModelScope/NVLM-D-72B).

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### Installation

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
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

## Deployment

### Single-node Deployment (4-NPU)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data
export MODEL_PATH="NVLM-D-72B"

vllm serve ${MODEL_PATH} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name NVLM-D-72B \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16
```

## Functional Verification

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "NVLM-D-72B",
        "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

Expected output:

```json
{"id":"chatcmpl-a17d171f4253ba40","object":"chat.completion","created":1779716499,"model":"NVLM-D-72B","choices":[{"index":0,"message":{"role":"assistant","content":"Large language models (LLMs) are a type of artificial intelligence model that are trained on vast amounts of text data. They use deep learning techniques, particularly transformer architectures, to understand and generate human-like text. These models can perform a wide range of tasks including text generation, translation, summarization, question answering, and code writing.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":12,"total_tokens":92,"completion_tokens":80,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

### Multimodal Verification (Image Understanding)

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "NVLM-D-72B",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_ENCODED_IMAGE>"}},
            {"type": "text", "text": "What is in this image? Describe briefly."}
        ]}],
        "max_tokens": 256,
        "temperature": 0.7
    }'
```

Expected output (with a real photo):

```json
{"choices": [{"message": {"content": "The image features an animated scene set in an outdoor environment during what appears to be either dawn or dusk, given the pastel hues of the sky..."}}], "usage": {"prompt_tokens": 823, "completion_tokens": 256}}
```
