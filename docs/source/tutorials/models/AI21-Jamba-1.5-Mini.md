# AI21-Jamba-1.5-Mini

## Introduction

AI21-Jamba-1.5-Mini is a compact Jamba-family hybrid model that combines Transformer and Mamba-style state space layers. The model supports instruction-style generation and long-context workloads, and it can be deployed on Ascend NPUs with `trust_remote_code` enabled.

This document provides a practical deployment and validation reference for `AI21-Jamba-1.5-Mini` on vLLM Ascend. The current validation baseline uses 2 NPUs, `bfloat16`, and ACLGraph for inference.

## Environment Preparation

### Installation

You can use the official docker image to run `AI21-Jamba-1.5-Mini` directly.

Refer to [using docker](../../installation.md#set-up-using-docker) for the container setup steps.

## Deployment

### Single-node Deployment (2-NPU)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

vllm serve "AI-ModelScope/AI21-Jamba-1.5-Mini" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name AI21-Jamba-1.5-Mini \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --trust-remote-code
```

### Deployment Notes

- `--trust-remote-code` is required for this model family.
- A 2-NPU deployment is recommended for the BF16 checkpoint on Atlas A2 Series hardware.
- ACLGraph is enabled by default when `--enforce-eager` is not set.

## Functional Verification

After the service starts, you can verify the deployment with the OpenAI-compatible endpoint:

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

- For larger-scale regression tracking, keep the same model path, precision, and NPU placement strategy when extending the evaluation.
