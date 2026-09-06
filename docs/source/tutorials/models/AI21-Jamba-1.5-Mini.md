# AI21-Jamba-1.5-Mini

## Introduction

AI21-Jamba-1.5-Mini is a compact Jamba-family hybrid model that combines Transformer and Mamba-style state space layers. It supports instruction-style generation and long-context workloads, and can be deployed on Ascend NPUs with `vllm-ascend` when `trust_remote_code` is enabled.

This document describes the main verification steps for `AI21-Jamba-1.5-Mini`, including environment preparation, single-node deployment, functional verification, and accuracy evaluation on GSM8K. The current validation baseline uses 2 NPUs, `bfloat16`, and ACLGraph for inference.

## Supported Features

Please refer to [Supported Models](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

## Environment Preparation

### Model Weight

- `AI21-Jamba-1.5-Mini` (BF16): requires about 2 Ascend 910B NPUs for the validated configuration. [Download model weight](https://www.modelscope.cn/models/AI-ModelScope/AI21-Jamba-1.5-Mini)

It is recommended to download the model weight into a shared cache directory such as `/root/.cache/` or a local path like `/data/models/AI21-Jamba-1.5-Mini`.

### Installation

You can use the official docker image to run `AI21-Jamba-1.5-Mini` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image).

```bash
# Update --device according to your device (Atlas A2: /dev/davinci[0-7]).
# Update the vllm-ascend image according to your environment.
# Note: download the weight to /root/.cache in advance.
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=1g \
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
    -it $IMAGE bash
```

If you prefer to build from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md).

## Deployment

### Single-node Deployment (2-NPU)

`AI21-Jamba-1.5-Mini` can be deployed on 1 Atlas A2 node with 2 NPUs for the BF16 checkpoint.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

vllm serve "AI-ModelScope/AI21-Jamba-1.5-Mini" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name AI21-Jamba-1.5-Mini \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

### Deployment Notes

- `--trust-remote-code` is required for this model family.
- A 2-NPU deployment is recommended for the BF16 checkpoint on Atlas A2 Series hardware.
- ACLGraph is enabled by default when `--enforce-eager` is not set.
- `--max-model-len` and `--gpu-memory-utilization` should stay consistent with the accuracy evaluation config.

## Functional Verification

After the service starts, verify the deployment with the OpenAI-compatible chat endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "AI21-Jamba-1.5-Mini",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to the Jamba architecture."}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

A valid response indicates that the model is deployed correctly.

## Accuracy Evaluation

The GSM8K dataset was used to evaluate the reasoning capability of `AI21-Jamba-1.5-Mini`.

The current evaluation setting is:

- Dataset: `gsm8k`
- Split: `test`
- Few-shot setting: `5-shot`
- `apply_chat_template`: `False`
- `fewshot_as_multiturn`: `False`

### Evaluation Command

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

lm_eval \
  --model vllm \
  --model_args pretrained=AI-ModelScope/AI21-Jamba-1.5-Mini,tensor_parallel_size=2,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.90 \
  --tasks gsm8k \
  --num_fewshot 5 \
  --apply_chat_template false \
  --fewshot_as_multiturn false \
  --batch_size auto
```

### Result

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Accuracy | gsm8k / test | exact_match,strict-match | 0.35 |
| Accuracy | gsm8k / test | exact_match,flexible-extract | 0.35 |

## Remarks

- Keep the same model path, precision, TP size, and NPU placement when extending accuracy or performance evaluation.
- For more evaluation tooling options, refer to [using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
