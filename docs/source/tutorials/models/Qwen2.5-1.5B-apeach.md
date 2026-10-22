# Qwen2.5-1.5B-apeach

## 1 Introduction

Qwen2.5-1.5B-apeach is a Korean hate-speech / harmful-content sequence classifier
based on the Qwen2.5-1.5B architecture (architectures: Qwen2ForSequenceClassification,
num_labels=2). It assigns each input text one of two labels:

- 0: Default (benign)
- 1: Spoiled (harmful)

The model card does not declare a third-party license; the base model
Qwen2.5-1.5B is Apache-2.0. This guide describes how to run the model with
vLLM Ascend for offline batch classification and online service deployment.

## 2 Supported Features

The model runs through the pooling runner with the `classify` task
(`supported_tasks=('classify', 'token_classify')`). Because it is a
classification model there is no sampling path: greedy/non-greedy criteria
do not apply, and inference is deterministic (the same input always yields
identical probabilities).

## 3 Prerequisites

### 3.1 Model Weight

The same weights are published on two sources:

- Hugging Face: [jason9693/Qwen2.5-1.5B-apeach](https://huggingface.co/jason9693/Qwen2.5-1.5B-apeach)
  (from mainland China you may need `HF_ENDPOINT=https://hf-mirror.com`)
- ModelScope: [Howeee/Qwen2.5-1.5B-apeach](https://www.modelscope.cn/models/Howeee/Qwen2.5-1.5B-apeach)

Download example (Hugging Face):

```shell
HF_ENDPOINT=https://hf-mirror.com hf download jason9693/Qwen2.5-1.5B-apeach
```

Download example (ModelScope):

```shell
modelscope download --model Howeee/Qwen2.5-1.5B-apeach
```

It is recommended to download the model weight to the shared directory of
multiple nodes, such as `/root/.cache/`.

## 4 Installation

Use the official docker image, refer to
[using docker](../../installation.md#set-up-using-docker), or build from
source, refer to [installation](../../installation.md).

## 5 Environment Configuration

**Primary (required for the custom op path):** the RMSNorm custom op
`aclnnAddRmsNormBias` is shipped in the `_cann_ops_custom` vendor package
and located via `ASCEND_CUSTOM_OPP_PATH`. `vllm_ascend_C.so` reads that
variable at load time, before Python-side `bootstrap_custom_op_env()` runs,
so it must be exported before starting vLLM:

```shell
. /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

This is the PR-validated configuration (verified with
`enable_custom_op() == True`).

**Fallback (only if the vendor script is unavailable):**
`VLLM_BATCH_INVARIANT=1` disables the custom op path
(`enable_custom_op()` returns `False`) and falls back to
`torch_npu.npu_add_rms_norm`. Note that this mode also makes
single-vs-batch outputs bit-exact, which the default custom op path does
not guarantee (~2e-3 kernel-level batch correlation is normal there).

## 6 Online Service Deployment

```shell
#!/bin/sh
# The custom op kernels (e.g. aclnnAddRmsNormBias) are shipped in the
# _cann_ops_custom vendor package and located via ASCEND_CUSTOM_OPP_PATH.
# vllm_ascend_C.so is loaded before Python-side bootstrap_custom_op_env()
# runs, so the variable must be exported before the process starts:
. /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

ASCEND_RT_VISIBLE_DEVICES=0 \
vllm serve /root/.cache/huggingface/hub/models--jason9693--Qwen2.5-1.5B-apeach/snapshots/<snapshot-sha> \
    --runner pooling \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enforce-eager \
    --port 8000
```

> Adjust the `set_env.bash` path if your vllm-ascend installation lives
> elsewhere; it exports `ASCEND_CUSTOM_OPP_PATH` and prepends the vendor
> `op_api/lib` directory to `LD_LIBRARY_PATH`.

Key parameter descriptions:

- `--runner pooling`: required for embedding/pooling/classification models
  (the legacy spelling is `--task classify`).
- `--dtype bfloat16`: bf16 keeps the probability tolerance vs the fp32
  reference within 1e-2 for confident inputs. Avoid fp16 on this model:
  a same-lineage fp16 test was historically flaky and is on the CI skip
  list.
- `--max-model-len 4096`: classification inputs are short (< 512 tokens),
  so the 128K context of the base model is unnecessary and only wastes KV
  cache.
- `--enforce-eager`: the model is not on the graph whitelist; eager mode is
  the validated configuration.

## 7 Functional Verification

Start the server as above, then classify raw text with the `/classify`
endpoint (classification models take plain text, no chat template):

```shell
curl http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/huggingface/hub/models--jason9693--Qwen2.5-1.5B-apeach/snapshots/<snapshot-sha>",
    "input": ["안녕하세요 오늘 날씨가 참 좋네요", "너 진짜 최악이다 다시는 보기 싶다"]
  }'
```

Expected result: HTTP 200 with per-input `probs` of length 2 (index 0 =
Default, index 1 = Spoiled). For the two samples above the first is
expected to be classified Default (high probability) and the second
Spoiled.

Offline batch classification example:

```python
from vllm import LLM

llm = LLM(
    model="/root/.cache/huggingface/hub/models--jason9693--Qwen2.5-1.5B-apeach/snapshots/<snapshot-sha>",
    runner="pooling",
    dtype="bfloat16",
    max_model_len=4096,
    enforce_eager=True,
)
outputs = llm.classify(["안녕하세요 오늘 날씨가 참 좋네요", "너 진짜 최악이다 다시는 보기 싶다"])
for out in outputs:
    print(out.outputs.probs)  # [P(Default), P(Spoiled)]
```

## 8 Known Limitations

- Architecture registration: `Qwen2ForSequenceClassification` is not in the
  vLLM model registry; vLLM resolves it through the transformers fallback
  backend (`TransformersForSequenceClassification`) automatically, so no
  `--hf-overrides` is needed.
- Custom op discovery: the RMSNorm custom op `aclnnAddRmsNormBias` is
  provided by the `_cann_ops_custom` vendor package, not by the CANN
  `libopapi.so` itself. `vllm_ascend_C.so` resolves it through
  `ASCEND_CUSTOM_OPP_PATH` at load time, so the vendor
  `bin/set_env.bash` must be sourced before starting vLLM; otherwise the
  first forward pass fails with `aclnnAddRmsNormBias ... not in
  libopapi.so`.
- Language coverage: the model is Korean-only. Chinese/English inputs run
  without crashing but are out of the model's domain.
- Soft probabilities: for benign Korean inputs the model naturally emits
  probabilities around 0.6-0.9, so bf16-vs-fp32 differences of ~0.01 are
  expected there; argmax labels remain stable.
