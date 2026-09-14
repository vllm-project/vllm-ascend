# apply_penalties

## Description

- **Function**: Applies the three vLLM sampling penalties — repetition, frequency and presence — to the sampling logits **in place**, on NPU. It replaces the `bincount` + `scatter` + elementwise torch chain of `vllm.model_executor.layers.utils.apply_penalties` with two Triton-Ascend kernels: `token_bin_counts_and_mask_kernel` (per-sequence token histogram and occurrence mask) and `apply_all_penalties_kernel` (the penalty update itself).
- **Formula**: For sequence `i` and vocabulary id `v`, with `prompt_mask` / `output_mask` marking whether `v` appeared in the prompt / in the generated output, and `output_bin_counts[i, v]` counting its occurrences in the generated output:
    - `repeated = prompt_mask[i, v] | output_mask[i, v]`
    - `p = repetition_penalties[i] if repeated else 1.0`
    - `logits[i, v] = logits[i, v] / p if logits[i, v] > 0 else logits[i, v] * p`
    - `logits[i, v] -= frequency_penalties[i] * output_bin_counts[i, v]`
    - `logits[i, v] -= presence_penalties[i] * output_mask[i, v]`
- **Algorithm flow** (processed row by row, independently):
  1. `get_token_bin_counts_and_mask_triton(prompt_tokens_tensor, vocab_size, num_seqs)` builds the prompt occurrence mask; the same helper is called again on `output_tokens_tensor` to build both the output mask and the output bin counts.
     - Grid: `grid_size = min(num_vectorcore, total_blocks)` where `total_blocks = num_seqs * ceil(seq_len / SEQ_BLOCK)` and `SEQ_BLOCK = 256`; each program walks its blocks with a grid-stride loop, which keeps the launch within the Triton-Ascend `coreDim` limit of 65535.
     - Per block: load `SEQ_BLOCK` token ids, keep only ids inside `[vocab_start_idx, vocab_start_idx + vocab_size)`, and `tl.atomic_add` 1 into the row histogram. Addresses of masked-out lanes are clamped to index 0 so no out-of-range address is formed. `vocab_start_idx = tp_rank * vocab_size` is non-zero only in reduce-sample (compressed-vocabulary) mode.
     - `mask = bin_counts > 0` is derived on the host side.
  2. `apply_all_penalties_kernel` updates the logits.
     - Grid: `grid = (min(num_seqs, num_vectorcore), 1, 1)`; each program handles `ceil(num_seqs / num_programs)` consecutive sequences.
     - Per sequence: load the three scalar penalties once, then sweep the vocabulary in `BLOCK_SIZE = 2048` tiles. For each tile, apply the repetition scaling (`1/p` for positive logits, `p` otherwise), subtract `frequency_penalty * output_bin_counts`, subtract `presence_penalty * output_mask`, and store back in place.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by `AscendSampler.apply_penalties` (`vllm_ascend/sample/sampler.py`) and by the rejection sampler (`vllm_ascend/sample/rejection_sampler.py`); it runs in the sampling stage after the model forward and is therefore outside the ACL graph capture region. When Triton is unavailable (`HAS_TRITON` is false), `AscendSampler` falls back to the default vLLM implementation.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input / Output | Sampling logits `[num_seqs, vocab_size]`, updated in place and also returned | fp32 / fp16 / bf16 | ND |
| `prompt_tokens_tensor` | Input | Prompt token ids `[num_seqs, max_prompt_len]`, right-padded with an id `>= vocab_size` | int64 | ND |
| `output_tokens_tensor` | Input | Generated token ids `[num_seqs, max_output_len]`, right-padded with an id `>= vocab_size` | int64 | ND |
| `presence_penalties` | Input | Per-sequence presence penalty `[num_seqs]` | fp32 | ND |
| `frequency_penalties` | Input | Per-sequence frequency penalty `[num_seqs]` | fp32 | ND |
| `repetition_penalties` | Input | Per-sequence repetition penalty `[num_seqs]` | fp32 | ND |

## Constraints

- `logits` must be 2-D `[num_seqs, vocab_size]`; `num_seqs` must match the first dimension of the two token tensors and the length of the three penalty tensors. The kernel reads the strides of every tensor, so non-contiguous `logits` is supported; `tokens` is made contiguous on the host side if needed.
- Padding contract: any token id `>= vocab_size` is ignored. Callers must pad with `vocab_size` (not `-1`); `vllm_ascend/sample/penalties.py` converts `-1` to `vocab_size` before the call. Negative ids other than `-1` are not handled.
- `max_prompt_len` or `max_output_len` equal to 0 is supported: `get_token_bin_counts_and_mask_triton` returns an all-zero histogram without launching a kernel.
- `num_seqs == 0` is **not** supported. The histogram helpers do return early, but `_apply_all_penalties_triton` still launches with `grid = (min(num_seqs, num_vectorcore), 1, 1)`, i.e. a zero-sized launch. Callers must skip the call for an empty batch; in serving the sampler is never invoked with zero sequences.
- `repetition_penalties` must be `> 0`; the kernel divides by it for positive logits. vLLM validates this at the request level (`repetition_penalty > 0`).
- The intermediate `bin_counts` is `int32` and is allocated as `[num_seqs, vocab_size]` per token tensor, so peak memory scales with `num_seqs * vocab_size * 4` bytes twice over.
- `BLOCK_SIZE` (2048) and `SEQ_BLOCK` (256) are compile-time `constexpr`; `vocab_size` and `seq_len` are handled by masked tiles, so no shape needs to be a multiple of them.
- `get_vectorcore_num()` must have been initialised: call `init_device_properties_triton()` once before the first invocation. In serving this happens during worker start-up; standalone callers must do it explicitly.
- The `AscendConfig` singleton must be initialised, because `get_token_bin_counts_and_mask_triton` reads `get_ascend_config().enable_reduce_sample` to decide whether `tp_rank` comes from the TP group or is fixed to 0. In serving this is done by `init_ascend_config()` during platform set-up; standalone callers must provide it themselves — the single-operator test calls `init_ascend_config(VllmConfig())` at import time, which is enough because single card always takes the `enable_reduce_sample = False` / `tp_rank = 0` path.
- Graph mode: not applicable — sampling runs outside the captured graph.

## Origin and Differences

- **Origin**: Migrated from `vllm.model_executor.layers.utils.apply_penalties` (equivalent to `vllm.v1.sample.ops.penalties.apply_all_penalties`), see <https://github.com/vllm-project/vllm-ascend/pull/6979>.
- **Differences**:
    - NPU adaptation for performance: the upstream implementation builds the histograms with `torch.scatter_add_` and then performs several full-vocabulary elementwise passes over the logits. Here the histogram is a single atomic-add Triton kernel and the three penalties are fused into one vocabulary sweep, so the `[num_seqs, vocab_size]` logits are read and written once instead of once per penalty. Work is distributed over vector cores, and the histogram kernel uses a 1-D grid-stride loop to stay within the Triton-Ascend `coreDim` limit of 65535 for long prompts.
    - Modified for a specific vllm-ascend logic or different input parameters: the histogram kernel additionally takes `tp_rank` and offsets the vocabulary window by `vocab_start_idx = tp_rank * vocab_size`, which supports the vllm-ascend reduce-sample (compressed-vocabulary) path where each TP rank holds only its own vocabulary shard. With `enable_reduce_sample` disabled — the default — `tp_rank` is 0 and the behaviour matches upstream.

## Test Cases

The accuracy test compares this operator against `vllm.v1.sample.ops.penalties.apply_all_penalties` element by element. It sweeps `num_seqs` in `{1, 8, 32, 128}` and `vocab_size` in `{5120, 151936}` (the latter being the Qwen3 vocabulary used in inference), over prompt/output length combinations that cover the empty, single-token, typical and all-padding cases, for both fp16 and bf16. The precision tolerance follows the data type: `rtol = atol = 1e-3` for fp16 and `1e-2` for bf16.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_penalties_triton.py
```
