# Speculative Decoding Guide

## Overview

Speculative decoding can reduce inter-token latency in memory-bound inference.
A proposer drafts tokens, and the target model verifies them in one decode step.
This guide covers Model Runner V2 on Ascend.

| `method` | Draft source |
| --- | --- |
| [`ngram_gpu`](#speculating-by-matching-n-grams-in-the-prompt) | Prompt n-gram matches on the NPU |
| [`eagle`, `eagle3`](#speculating-using-eagle-based-draft-models) | A matching EAGLE or EAGLE-3 checkpoint |
| [`mtp`](#speculating-using-mtp) | Multi-Token Prediction weights built into the target or supplied by a matching assistant checkpoint |
| [`dflash`](#speculating-using-dflash) | A DFlash, [DFlash2](#speculating-using-dflash2), or [LiLiCorr](#speculating-using-lilicorr) checkpoint |
| [`dspark`](#speculating-using-dspark) | A matching DSpark checkpoint |
| [`draft_model`](#speculating-using-a-draft-model) | A compatible standalone draft LLM with the same vocabulary size |
| [`extract_hidden_states`](#extracting-hidden-states) | Hidden-state collection for draft training |

## Common Configuration {: #common-configuration }

Use `LLM(speculative_config=...)` for offline inference or pass the same settings
as JSON to `vllm serve --speculative-config` (`-sc`) for online serving. The
examples below include a launch command for each drafting method. Use local
checkpoint paths when you already have the weights downloaded.

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    speculative_config={
        "method": "ngram_gpu",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 5,
    },
)
outputs = llm.generate(["The future of AI is"])
```

```shell
vllm serve Qwen/Qwen3-1.7B \
  --speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

- `num_speculative_tokens` sets the draft length `K`. Some checkpoints supply
    it through `n_predict`. On paths that use `npu_fused_infer_attention_score`,
    keep `K + 1` at or below 16.
- Set `model` inside `speculative_config` for methods that load a separate
    draft checkpoint. Use a local path or a checkpoint ID.
- Set `draft_tensor_parallel_size` to 1 or the target TP size, subject to the
    drafter's requirements.
- For fixed-length full ACLGraph with EAGLE, DFlash2, or MTP with `K > 1`, use
    multiples of `K + 1` when setting capture sizes manually.
- For prefill/decode (P/D) separation, use the same `K` on both roles for
    hybrid Mamba models. Use the model tutorial for role-specific serving and
    parallelism settings. See the [model tutorials](../../tutorials/models/index.md).

## Speculating by matching n-grams in the prompt {: #speculating-by-matching-n-grams-in-the-prompt }

With `ngram_gpu`, set `prompt_lookup_min` and `prompt_lookup_max` to bound the
n-gram lengths to search. See the [offline and online examples](#common-configuration).

## Speculating using EAGLE based draft models {: #speculating-using-eagle-based-draft-models }

Set `method` to `eagle` or `eagle3` according to the draft checkpoint. Set
`model` to the matching draft and use `draft_tensor_parallel_size=1` for EAGLE.

EAGLE with ModelScope checkpoints:

```shell
VLLM_USE_MODELSCOPE=True vllm serve LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method":"eagle",
    "model":"vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens":3,
    "draft_tensor_parallel_size":1
  }'
```

EAGLE-3:

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
```

## Speculating using MTP {: #speculating-using-mtp }

Set `method` to `mtp`. For a target with built-in MTP weights, omit `model`
from the speculative config. Start with `K=1` and measure acceptance and
throughput before increasing it, especially with a single prediction layer.

```shell
vllm serve Qwen/Qwen3.5-4B \
  --dtype float16 \
  --mamba-ssm-cache-dtype float16 \
  --max-model-len 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For Gemma 4 with an assistant, set `model` to the matching assistant checkpoint
and keep `method="mtp"`. This example uses two A5 NPUs.

```shell
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}'
```

## Speculating using DFlash {: #speculating-using-dflash }

Set `method` to `dflash` and `model` to a checkpoint trained for the target.
The checkpoint's architecture selects the DFlash variant.

```shell
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16","num_speculative_tokens":7}'
```

### Speculating using DFlash2 {: #speculating-using-dflash2 }

<span id="how-it-works"></span>Use a `DFlash2DraftModel` checkpoint and the
target's unquantized, full-vocabulary LM head. The checkpoint supplies the
candidate selector settings and supported draft lengths.

Draft sampling is greedy by default. Set `draft_sample_method` to
`probabilistic` to sample at the request temperature. Full draft ACLGraph
capture requires target `FULL_DECODE_ONLY` or `FULL_AND_PIECEWISE` and draft
`enforce_eager` unset or false.

```shell
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16]}' \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

### Speculating using LiLiCorr {: #speculating-using-lilicorr }

Use a matching `LiLiCorrDraftModel` checkpoint and set `"enforce_eager":true`
inside `speculative_config`. Set `K` at or below the trained block size minus
one. Both plain and convolutional checkpoints support greedy and probabilistic
drafting with standard TP. Fine-grained embedding or LM-head TP and PCP
vocabulary sharding are unsupported. Quantized draft weights must use an
Ascend-supported format. The example uses a draft checkpoint from ModelScope.

```shell
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{
    "method":"dflash",
    "model":"niannianxu/Qwen3.8-27B-LiLiCorr-block8-epoch0",
    "num_speculative_tokens":7,
    "draft_tensor_parallel_size":2,
    "enforce_eager":true
  }'
```

## Speculating using DSpark {: #speculating-using-dspark }

Set `method` to `dspark`, `model` to a matching draft checkpoint, and `K` to
a length supported by that checkpoint. If an MLA draft uses BF16 while the
target uses FP8 KV cache, set `"kv_cache_dtype":"bfloat16"` in
`speculative_config`.

```shell
vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","num_speculative_tokens":7}'
```

### Adaptive verification

With a DSpark checkpoint that has a confidence head, set
`"enable_adaptive_verification":true` to select a verification length per
request. Enable full decode graphs on the target. PP, PCP, DCP, and LoRA
are unsupported, and commercial use is restricted.

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

#### Hybrid Gated DeltaNet targets

Adaptive verification on hybrid GDN targets requires triton-ascend and is
unsupported on 310P. Compare fixed and adaptive DSpark at the same workload
and concurrency. See the [upstream adaptive verification guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/).

## Speculating using a draft model {: #speculating-using-a-draft-model }

Set `method` to `draft_model` and `model` to a compatible standalone draft.

```shell
vllm serve Qwen/Qwen3-8B \
  --enforce-eager \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

## Extracting Hidden States {: #extracting-hidden-states }

Use `extract_hidden_states` with `num_speculative_tokens=1` to collect training
data for EAGLE-style drafts. Select layers with
`draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids` and use
`ExampleHiddenStatesConnector` with `kv_role="kv_producer"`. The connector
writes one `.safetensors` file per request to `shared_storage_path`. See the
[upstream offline example](https://docs.vllm.ai/en/latest/features/speculative_decoding/extract_hidden_states/#offline-example).

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

## Dynamic Speculative Decoding {: #dynamic-speculative-decoding }

### Batch-size based (Autoregressive) {: #batch-size-based-autoregressive }

<span id="why-is-it-needed"></span><span id="use-cases"></span><span id="-speculative-config-schema"></span>Add `num_speculative_tokens_per_batch_size` to the existing proposer config.
Each entry is an inclusive `[min_batch_size, max_batch_size, draft_length]`
range. A draft length of 0 disables drafting for that range.

<span id="online-examples"></span><span id="dynamic-sd-eagle-drafter"></span><span id="dynamic-sd-eagle3-drafter"></span>The following settings use three, one, or zero draft tokens as the batch grows.

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

Use DP=1 and DCP=1. With DP enabled, the engine falls back to a fixed draft
length. Ascend rejects DCP when the dynamic schedule is active.

## Block Verification

Set `"rejection_sample_method":"block"` to verify a block of draft tokens
jointly while preserving the target sampling distribution. Greedy requests
retain greedy verification. Use Block Verification separately from Entropy
Verify and synthetic rejection sampling.

## Entropy Verify {: #entropy-verify }

<span id="usage"></span>Entropy Verify changes the acceptance threshold for
non-greedy requests. It can improve throughput and change the output. Evaluate
output quality on your workload. Use `"rejection_sample_method":"standard"`
and enable Entropy Verify through `--additional-config`.

```shell
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3,"rejection_sample_method":"standard"}' \
  --additional-config '{"rejection_sampler_config":{"enable_entropy_verify":true}}'
```

| Option | Default | Effect |
| --- | --- | --- |
| `posterior_threshold` | `0.95` | Caps the threshold. Valid range is `(0, 1]`. |
| `posterior_alpha` | `0.4` | Controls the entropy adjustment. Use a nonnegative value or 0 for a fixed threshold. |

The threshold per position is `min(exp(-entropy * posterior_alpha), posterior_threshold)`,
using entropy computed after request sampling settings.

## Synthetic Rejection Sampling

<span id="why-use-it"></span>Set `"rejection_sample_method":"synthetic"` for
controlled benchmarks and validation only. It can change generated text.

<span id="how-to-use"></span>Set exactly one of these fields alongside the proposer settings.

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

<span id="ascend-specific-notes"></span>

## RecoverSSM compatibility

<span id="experimental-qwen-gdn-recoverssm"></span>On A3 and A5, enable
experimental recurrent-state reconstruction for Qwen3.x Gated DeltaNet (GDN)
models with `--use-replayssm`. Use MTP, DSpark, or DFlash with PP=PCP=DCP=1,
BF16 activations and convolution state, and BF16 or FP32 SSM state.
Disable stochastic rounding.

Use Mamba cache mode `align` with prefix caching and `none` without it.
In `align` mode, use a Mamba block size of at least `K + 1`.

For P/D, enable RecoverSSM on both roles and match the speculative window,
state dtypes, convolution layout, and prefix-caching settings. Use
`MooncakePullConnector` or `MooncakeConnectorV2` in pull mode
(`use_layerwise=false`).
