# IQuest-Coder-V1-40B-Instruct

## Introduction

IQuest-Coder-V1 is a family of code large language models built on a LLaMA-style
architecture with Grouped-Query Attention (GQA). The 40B-Instruct variant has
80 layers, `hidden_size=5120`, `num_attention_heads=40`, `num_key_value_heads=8`,
`head_dim=128`, `intermediate_size=27648`, `vocab_size=76800` and a 128K token
context (`max_position_embeddings=131072`). Its weights use the standard LLaMA
tensor naming, and the optional OLMo (`clip_qkv`) / Qwen2 (`sliding_window`)
enhancements are **disabled** in this config, so inference behaviour is identical
to a standard LLaMA + GQA model.

This tutorial shows how to run IQuest-Coder-V1-40B-Instruct on Ascend NPUs with
vLLM-Ascend. It requires **vllm-ascend >= 0.18.0rc1**.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md)
to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the
feature's configuration.

## Environment Preparation

### Model Weight

- `IQuest-Coder-V1-40B-Instruct` (BF16, ~80 GB): requires **2 × Atlas 800 A2 (64 GB)**
  cards for tensor-parallel-size = 2. A single 64 GB card cannot hold the ~80 GB
  of weights, so **single-card deployment is not supported** for the BF16 model.
  [Download model weight](https://modelscope.cn/models/IQuestLab/IQuest-Coder-V1-40B-Instruct)

It is recommended to download the model weight (via the ModelScope or
HuggingFace repo linked above) to a local directory of your choice.

### NPU device selection

Ascend runtime logical device IDs are controlled via `ASCEND_RT_VISIBLE_DEVICES`.
For example, to restrict the process to the last four physical NPUs (4, 5, 6, 7):

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

> Note: the runtime renumbers the listed physical devices to logical 0..N-1.
> `tensor_parallel_size` must divide both `num_attention_heads` (40) and
> `num_key_value_heads` (8), so the valid values are **1 / 2 / 4 / 8**. With the
> last-four cards you can use TP=2 (or TP=4 if all four are present).

### Verify Multi-node Communication (Optional)

If you want to deploy a multi-node environment, verify multi-node communication
according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

### Installation

You can use our official docker image for supporting IQuest-Coder-V1-40B-Instruct.
Currently, we provide the all-in-one images.
[Download images](https://quay.io/repository/ascend/vllm-ascend?tab=tags)

#### Docker Pull (by tag)

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|

```

#### Docker run

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7]).
# Update the vllm-ascend image according to your environment.
# Download the model weights to a local directory before running.
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --entrypoint /bin/bash \
    $IMAGE
```

## Quick Start

### 1. Architecture registration (already in vllm-ascend)

IQuest-Coder-V1-40B-Instruct declares a custom architecture name
`IQuestCoderForCausalLM` that is not part of vLLM's built-in registry. vllm-ascend
ships a platform patch
(`vllm_ascend/patch/platform/patch_iquestcoder_config.py`) that aliases it to
`LlamaForCausalLM`. After importing vllm-ascend the model loads natively — **no
`--hf-overrides` needed**.

If you are on an older vllm-ascend without the patch, you can still force the
mapping with:

```bash
--hf-overrides '{"architectures":["LlamaForCausalLM"],"model_type":"llama"}'
```

### 2. Start the OpenAI-compatible server

```bash
# Use 2 NPUs (TP=2) for the 40B BF16 model.
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

vllm serve IQuestLab/IQuest-Coder-V1-40B-Instruct \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

For a local weight path, replace the model id with the directory path where
you saved the weights.

### 3. Query the server

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "IQuestLab/IQuest-Coder-V1-40B-Instruct",
      "prompt": "def quicksort(arr):",
      "max_tokens": 128,
      "temperature": 0.0
    }'
```

### 4. Verified behaviour

On Atlas 800 A2 (910B3), with the settings above:

- **Eager mode**: ~22 s to generate a short code sample; model loads in ~70 s,
  ~37 GB weights per card, ~114 K tokens of KV cache at `max-model-len=2048`.
- **ACL graph mode**: same workload generates in ~4 s (graph replay captures
  LlamaForCausalLM, 80 layers).

Both modes produce correct code and correct answers (e.g. `O(log n)` for binary
search complexity).

> Note: vLLM reports `Using a slow tokenizer` because the model ships a non-fast
> `IQuestCoderTokenizer`. This is benign; generation quality is unaffected.

## Accuracy and Performance Evaluation

The e2e accuracy config is at
`tests/e2e/models/configs/IQuest-Coder-V1-40B-Instruct.yaml`. Run it with:

```bash
pytest tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/IQuest-Coder-V1-40B-Instruct.yaml \
    --tp-size 2
```

It evaluates `humaneval` and `mbpp` (pass@1) on Ascend. The metric thresholds in
the yaml are preliminary estimates for the Instruct variant on bf16; the nightly
accuracy run confirms the final numbers.

## Known Limitations

- **Single-card BF16 is unsupported** — the 40B weights (~80 GB) exceed a single
  64 GB card. Use TP>=2 (or a quantized W8A8 variant for single-card).
- **Long context KV cache is large**: at 128 K tokens, KV cache per card is
  substantial; reduce `--max-model-len` for higher concurrency on 2 cards.
- **Slow tokenizer**: convert to a fast tokenizer if throughput is critical.
