# Speculative Decoding Guide

This guide shows how to use Speculative Decoding with vLLM Ascend. Speculative decoding is a technique which improves inter-token latency in memory-bound LLM inference.

## Overview

vLLM Ascend implements speculative decoding through a **speculator-verifier** architecture.

1. The **Speculator** generates draft (speculative) tokens using various methods, from simple n-gram matching to neural-network-based draft models.
2. The **Rejection Sampler** verifies draft tokens against the target model's output, accepting matches and rejecting mismatches, with optional block verification configured through `speculative_config.rejection_sample_method`.

## Quickstart

For offline inference, set `speculative_config` in `LLM()`. For online serving, pass the same
settings as JSON to `vllm serve --speculative-config` (`-sc`). Each example below
shows both forms. Use local checkpoint paths when you already have the weights.

- `num_speculative_tokens` sets the draft length `K`. Some checkpoints supply
    it through `n_predict`.
- Set `model` inside `speculative_config` for methods that load a separate
    draft checkpoint. Use a local path or a checkpoint ID.
- Set `draft_tensor_parallel_size` to 1 or the target TP size, subject to the
    drafter's requirements.
- For prefill/decode (P/D) separation, use the same `K` on both roles for
    hybrid Mamba models. Use the model tutorial for role-specific serving and
    parallelism settings. See the [model tutorials](../../tutorials/models/index.md).

## Speculative Decoding Methods

| `method` | Draft source or purpose |
| --- | --- |
| [`ngram_gpu`](#n-gram) | Prompt n-gram matches on the NPU |
| [`eagle`](#eagle) | A matching EAGLE checkpoint |
| [`eagle3`](#eagle-3) | A matching EAGLE-3 checkpoint |
| [`mtp`](#mtp) | Multi-Token Prediction weights built into the target or supplied by a matching assistant checkpoint |
| [`dflash`](#dflash) | A DFlash, [DFlash2](#dflash2), or [LiLiCorr](#lilicorr) checkpoint |
| [`dspark`](#dspark) | A matching DSpark checkpoint |
| [`draft_model`](#standalone-draft-model) | A compatible standalone draft LLM with the same vocabulary size |
| [`extract_hidden_states`](#hidden-state-extraction) | Target hidden states for draft-model training |

### N-gram

With `ngram_gpu`, set `prompt_lookup_min` and `prompt_lookup_max` to bound the
n-gram lengths to search.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    speculative_config={
        "method": "ngram_gpu",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 5,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-1.7B \
  --speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

### EAGLE

Set `method` to `eagle`, `model` to the matching draft checkpoint, and
`draft_tensor_parallel_size` to 1. The example uses ModelScope checkpoints.

**Offline inference**

```python
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
from vllm import LLM, SamplingParams

llm = LLM(
    model="LLM-Research/Meta-Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
VLLM_USE_MODELSCOPE=True vllm serve LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method":"eagle",
    "model":"vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens":3,
    "draft_tensor_parallel_size":1
  }'
```

### EAGLE-3

Set `method` to `eagle3` and `model` to the matching draft checkpoint.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
```

### MTP

Set `method` to `mtp`. For a target with built-in MTP weights, omit `model`
from the speculative config.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-4B",
    dtype="float16",
    mamba_ssm_cache_dtype="float16",
    max_model_len=4096,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3.5-4B \
  --dtype float16 \
  --mamba-ssm-cache-dtype float16 \
  --max-model-len 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For a target that uses an assistant, such as Gemma 4, set `model` in the
speculative config to the matching assistant checkpoint. See the
[Gemma 4 tutorial](../../tutorials/models/Gemma4.md) for deployment.

!!! note "Single-layer MTP"

    DeepSeek checkpoints with one MTP prediction layer reuse it for longer
    drafts. Start with `K=1` and check accuracy and throughput before increasing it.

### DFlash

Set `method` to `dflash` and `model` to a checkpoint trained for the target.
The checkpoint's architecture selects the DFlash variant.

#### DFlash2

Use a `DFlash2DraftModel` checkpoint and the target's unquantized,
full-vocabulary LM head. The checkpoint supplies the candidate selector
settings and supported draft lengths.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.8-27B",
    tensor_parallel_size=2,
    max_model_len=2048,
    enable_prefix_caching=False,
    compilation_config={
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [8, 16],
    },
    speculative_config={
        "method": "dflash",
        "model": "z-lab/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16]}' \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

Full draft ACLGraph capture requires target `FULL_DECODE_ONLY` or
`FULL_AND_PIECEWISE` and draft `enforce_eager` unset or false.

#### LiLiCorr

Use a matching `LiLiCorrDraftModel` checkpoint and set `K` at or below the
trained block size minus one. Both plain and convolutional checkpoints support
greedy and probabilistic drafting with standard TP.

Set `"enforce_eager":true` inside `speculative_config` for the draft.
Fine-grained embedding or LM-head TP and PCP vocabulary sharding are
unsupported. Quantized draft weights require an Ascend-supported format.

### DSpark

Set `method` to `dspark`, `model` to a matching draft checkpoint, and `K` to
a length supported by that checkpoint. If an MLA draft uses BF16 while the
target uses FP8 KV cache, set `"kv_cache_dtype":"bfloat16"` in
`speculative_config`.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
    },
    speculative_config={
        "method": "dspark",
        "model": "deepseek-ai/dspark_qwen3_8b_block7",
        "num_speculative_tokens": 7,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","num_speculative_tokens":7}'
```

On 950DT, GLM with a compatible GQA draft can use CANN FlashAttn.
Compatible MLA drafts can use FlashMLA. Both paths require FP16 or BF16
queries and an unquantized draft KV cache. FlashAttn pads 192-wide GQA heads
to 256, using one third more draft KV cache memory for those heads. MLA
drafts fall back to FIA if the FlashMLA query-head layout is unsupported.

For per-request verification lengths, see [Adaptive Verification](#adaptive-verification).

### Standalone Draft Model

Set `method` to `draft_model` and `model` to a compatible standalone draft.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    enable_prefix_caching=False,
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 3,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

### Hidden State Extraction

Use `extract_hidden_states` with `num_speculative_tokens=1` to collect training
data for EAGLE-style drafts. Select layers with
`draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids` and use
`ExampleHiddenStatesConnector` with `kv_role="kv_producer"`. The connector
writes one `.safetensors` file per request to `shared_storage_path`. See the
[upstream offline example](https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states/#offline-example).

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [2, 18, 34],
            },
        },
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/dev/shm/hidden_states",
        },
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].kv_transfer_params["hidden_states_path"])
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{
    "method":"extract_hidden_states",
    "num_speculative_tokens":1,
    "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[2,18,34]}}
  }' \
  --kv-transfer-config '{
    "kv_connector":"ExampleHiddenStatesConnector",
    "kv_role":"kv_producer",
    "kv_connector_extra_config":{"shared_storage_path":"/dev/shm/hidden_states"}
  }'
```

!!! note

    For prompt-only collection, request one output token with `max_tokens=1`.
    The connector saves prompt hidden states by default. Set
    `include_output_tokens` in the request's `kv_transfer_params` to collect
    generated-token hidden states as well.

## Sampling Configuration

### Draft Sampling

Draft sampling is greedy by default. For drafters that support probabilistic
sampling, add this to `speculative_config` to sample at the request temperature.

```json
{
  "draft_sample_method": "probabilistic"
}
```

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "draft_sample_method": "probabilistic",
    },
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "draft_sample_method": "probabilistic"
  }'
```

### Block Verification

Add this to `speculative_config` to verify draft tokens as a block while
preserving the target sampling distribution.

```json
{
  "rejection_sample_method": "block"
}
```

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "block",
    },
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "rejection_sample_method": "block"
  }'
```

Greedy requests retain greedy verification. Use Block Verification separately
from Entropy Verify and synthetic rejection sampling.

### Entropy Verify

Entropy Verify changes the acceptance threshold for
non-greedy requests to improve throughput.

Keep `rejection_sample_method` at its default, `"standard"`, and add this to
`additional_config` (`--additional-config` for online serving).

```json
{
  "rejection_sampler_config": {
    "enable_entropy_verify": true
  }
}
```

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "standard",
    },
    additional_config={
        "rejection_sampler_config": {
            "enable_entropy_verify": True,
        },
    },
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "rejection_sample_method": "standard"
  }' \
  --additional-config '{
    "rejection_sampler_config": {
      "enable_entropy_verify": true
    }
  }'
```

| Option | Default | Effect |
| --- | --- | --- |
| `posterior_threshold` | `0.95` | Caps the threshold. Valid range is `(0, 1]`. |
| `posterior_alpha` | `0.4` | Controls the entropy adjustment. Use a nonnegative value or 0 for a fixed threshold. |

The threshold per position is `min(exp(-entropy * posterior_alpha), posterior_threshold)`,
using entropy computed after request sampling settings.

!!! warning

    Changing the acceptance threshold can change generated tokens. Evaluate
    output quality on your workload before using Entropy Verify in production.

### Synthetic Rejection Sampling

Set `"rejection_sample_method":"synthetic"` to
benchmark the verifier at controlled acceptance rates.

Set exactly one of these fields in `speculative_config`.

- `synthetic_acceptance_rates` is a non-increasing list of `K` probabilities
    in `[0, 1]`. The probability at position `i` covers accepting the prefix
    through that position.
- `synthetic_acceptance_length` is the target mean acceptance length in
    `[1, K + 1]`.

For `K=3`, add these fields to your existing `speculative_config`.

```json
{
  "rejection_sample_method": "synthetic",
  "synthetic_acceptance_rates": [0.9, 0.6, 0.3]
}
```

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rates": [0.9, 0.6, 0.3],
    },
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "rejection_sample_method": "synthetic",
    "synthetic_acceptance_rates": [
      0.9,
      0.6,
      0.3
    ]
  }'
```

!!! warning

    Synthetic mode accepts draft tokens according to configured rates,
    independently of the target distribution. Use it only for benchmarks and
    validation.

#### Ascend-specific Notes

Ascend supports synthetic acceptance for both greedy and non-greedy
requests. The NPU rejection kernel draws uniform values with FP32 Triton
`tl.rand` and clamps them away from zero. The configuration fields match
upstream vLLM.

## Other Speculative Decoding Settings

### Dynamic Speculative Decoding

Add `num_speculative_tokens_per_batch_size` to `speculative_config`.
Each entry is an inclusive `[min_batch_size, max_batch_size, draft_length]`
range. A draft length of 0 disables drafting for that range.

The following settings use three, one, or zero draft tokens as the batch grows.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    enforce_eager=True,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "num_speculative_tokens": 3,
        "num_speculative_tokens_per_batch_size": [
            [1, 64, 3],
            [65, 128, 1],
            [129, 512, 0],
        ],
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{
    "method":"eagle3",
    "model":"RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens":3,
    "num_speculative_tokens_per_batch_size":[[1,64,3],[65,128,1],[129,512,0]]
  }'
```

Use DP=1 and DCP=1. With DP greater than 1, the engine falls back to a fixed
draft length. Ascend rejects DCP greater than 1 when the dynamic schedule is
active.

### Adaptive Verification

With a DSpark checkpoint that has a confidence head, set
`"enable_adaptive_verification":true` to select a verification length per
request.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
    },
    speculative_config={
        "method": "dspark",
        "model": "deepseek-ai/dspark_qwen3_8b_block7",
        "num_speculative_tokens": 7,
        "enable_adaptive_verification": True,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{
    "method":"dspark",
    "model":"deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens":7,
    "enable_adaptive_verification":true
  }'
```

!!! warning

    Enable full decode ACLGraphs on the target. Fully eager and pure
    `PIECEWISE` modes are unsupported, as are PP, PCP, DCP, and LoRA.
    Commercial use is restricted.

Compare fixed and adaptive verification at your serving concurrency.
Adaptive verification can add latency in latency-sensitive or host-bound
workloads. Gains can be limited when draft acceptance is already high.
Check scheduled and admitted draft counts in DEBUG logs when diagnosing
performance.

#### Hybrid Gated DeltaNet Targets

Adaptive verification on hybrid GDN targets requires triton-ascend and is
unsupported on 310P. See the
[upstream adaptive verification guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/).

### RecoverSSM

RecoverSSM reconstructs
the accepted recurrent state from compact replay records during speculative
decoding. This substantially reduces NPU memory use and can slightly speed up
Mamba layers in compute-bound workloads. The examples below use
Qwen3.6-35B-A3B with MTP.

**Offline inference**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.6-35B-A3B",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=4096,
    enable_prefix_caching=False,
    mamba_cache_mode="none",
    mamba_ssm_cache_dtype="float32",
    use_replayssm=True,
    enforce_eager=True,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 4,
    },
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The future of AI is"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Online serving**

```shell
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --mamba-ssm-cache-dtype float32 \
  --use-replayssm \
  --enforce-eager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

!!! warning

    RecoverSSM is experimental. Validate accuracy and performance on your
    workload before deployment. Enable speculative decoding and disable
    stochastic rounding. With prefix caching, use Mamba cache mode `align`
    and a block size of at least `K + 1`. Without prefix caching, use `none`.

    For direct prefill/decode (P/D) transfer, use `MooncakePullConnector`.
    Enable RecoverSSM on both roles and match the model weights, speculative
    method and window, draft checkpoint if used, prefix-caching settings,
    state dtypes, and convolution layout.
