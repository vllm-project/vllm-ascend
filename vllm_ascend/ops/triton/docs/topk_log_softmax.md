# topk_log_softmax

> Source: `vllm_ascend/worker/v2/sample/logprob.py` (`_topk_log_softmax_kernel`, exposed as `compute_token_logprobs`).

## Description

- **Function**: Computes the log-probability of a selected set of token ids per request. It fuses `log_softmax` over the whole vocabulary with the gather at the requested token positions, so the `[num_reqs, vocab_size]` intermediate that a `log_softmax` + `gather` pair would materialize is never written to global memory. It backs the model-runner-v2 sampler logprob path (`compute_token_logprobs`) and, through `compute_topk_logprobs`, the sampler / prompt-logprob / rejection-sampler logprob outputs.
- **Formula**: for request `i` and slot `k`, with `t = token_ids[i, k]`:
    - `m_i = max_j logits[i, j]`
    - `lse_i = log(sum_j exp(logits[i, j] - m_i))`
    - `output[i, k] = logits[i, t] - lse_i - m_i`

  The max subtraction is the standard numerically stable form: the reduction is accumulated in fp32 even when `logits` is fp16/bf16, so `exp` never overflows for the value ranges produced by a language-model head.
- **Algorithm flow** (processed row by row, independently):
  1. Grid is `(num_reqs,)`: one program per request, each owning one full logits row (`row_ptr = logits_ptr + req_idx * logits_stride`).
  2. Online normalization: load each vocabulary tile once, cast to fp32, and update `new_max = max(max_val, max(tile))`. Rescale the previous exponential sum by `exp(max_val - new_max)` before adding `sum(exp(tile - new_max))`.
  3. Fully masked prefixes: use zero as the temporary subtraction reference while the running maximum is `-inf`, so those tiles contribute zero rather than NaN. After the final tile, take `lse_i = log(se)`. An entirely `-inf` row still produces NaN logprobs, matching the reference.
  4. Gather: load `PADDED_TOPK` token ids for the request under the mask `k_offset < topk`, gather `logits[i, t]` at those ids, compute `logits - lse_i - m_i` in fp32 and store the `topk` valid lanes.

  The normalization reads the row once from global memory. UB usage remains bounded by `BLOCK_SIZE` rather than `vocab_size`; the launcher retains `multibuffer=False`.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. Used by vLLM model-runner-v2 sampling; it is installed over the upstream implementation in `vllm_ascend/patch/worker/patch_triton.py`. It runs in the sampler post-processing path, i.e. eagerly, outside ACL Graph capture.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` (`logits_ptr`) | Input | Raw logits of the LM head, `[num_reqs, vocab_size]` | fp32 / fp16 / bf16 | ND |
| `token_ids` (`topk_ids_ptr`) | Input | Token ids whose logprob is requested, `[num_reqs, topk]`; cast to int64 by the wrapper | int32 / int64 | ND |
| `logits_stride` | Input (attribute) | Row stride of `logits`, i.e. `logits.stride(0)` | int32 | scalar |
| `topk` | Input (attribute) | Number of token ids per request (`token_ids.shape[1]`) | int32 | scalar |
| `vocab_size` | Input (attribute) | Vocabulary size, i.e. `logits.shape[1]` | int32 | scalar |
| `BLOCK_SIZE` | Attribute (`tl.constexpr`) | Vocabulary tile length of the online reduction; pinned to `15 * 1024` by `compute_token_logprobs` | int32 | scalar |
| `PADDED_TOPK` | Attribute (`tl.constexpr`) | Lane count of the gather, `max(next_power_of_2(topk), 2)` | int32 | scalar |
| `logprobs` (`output_ptr`) | Output | Log probabilities of the requested tokens, `[num_reqs, topk]` | fp32 | ND |

## Constraints

- `logits`: 2-D `[num_reqs, vocab_size]`, fp32 / fp16 / bf16. `vocab_size` is a runtime argument and needs no alignment — the tail of each `BLOCK_SIZE` tile is masked with `-inf` — but rows are addressed through `logits_stride`, so the last row must be fully materialized (a view whose final row is truncated is not supported). `num_reqs` equals the grid size, so parallelism scales with the batch, not with the vocabulary: a single-request call uses one vector core regardless of `vocab_size`.
- `token_ids`: `[num_reqs, topk]`, integer, values in `[0, vocab_size)`. The kernel performs no bounds check; an out-of-range id reads out of bounds. The row stride is passed explicitly, supporting broadcast rows and padded row storage; the last dimension must remain contiguous. Output rows are contiguous.
- `topk >= 1`. `PADDED_TOPK` is `constexpr`, so every distinct `next_power_of_2(topk)` costs one kernel recompilation; `topk = 0` is handled by the caller (`compute_topk_logprobs` always concatenates the sampled token, giving at least one column).
- `logprobs` is always fp32, independent of the `logits` dtype; the reduction is accumulated in fp32 as well.
- `BLOCK_SIZE = 15 * 1024` was evaluated on A2 with Triton-Ascend 3.6. Raising it, or reintroducing multi-buffering, requires checking the UB budget. The `propagate_nan=tl.PropagateNan.ALL` form of `tl.maximum` preserves NaN handling (see [#9193](https://github.com/vllm-project/vllm-ascend/pull/9193)).
- Graph mode: not captured. Sampling and logprob computation run eagerly after the captured model forward, so the kernel sees dynamic `num_reqs` on every call; only `BLOCK_SIZE` and `PADDED_TOPK` are compile-time constants.
- Inference only: there is no backward pass.

## Origin and Differences

- **Origin**: adapted from vLLM's GPU implementation `vllm/v1/worker/gpu/sample/logprob.py` (recorded in the file header). The Ascend version is installed over the upstream symbols by `vllm_ascend/patch/worker/patch_triton.py` (`logprob.compute_token_logprobs`, and `compute_topk_logprobs` for `sampler`, `prompt_logprob` and `rejection_sampler`).
- **Differences**:
    - NPU adaptation for performance: online normalization uses `BLOCK_SIZE = 15 * 1024` with `multibuffer=False`; `tl.maximum` retains `propagate_nan=tl.PropagateNan.ALL`. The companion `_ranks_kernel` used by `compute_topk_logprobs` is parallelized over `get_vectorcore_num()` cores with a `rows_per_core` row loop (and `do_not_specialize` on `batch_size` / `rows_per_core`) instead of one program per request, because an NPU has far fewer, much wider cores than a GPU has SMs.
    - Modified for a specific vllm-ascend logic or different input parameters: `compute_topk_logprobs` builds `logprob_token_ids` with `torch.topk` + `torch.cat` and then calls this kernel once for the concatenated ids. The upstream `logprob_token_ids_state` / `expanded_idx_mapping` / `max_per_req_token_ids` path is accepted for signature compatibility but not implemented — a non-zero `max_per_req_token_ids` logs a warn-once and falls back to the default token ids.

## Test Cases

Accuracy tests live under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`:

- `test_compute_token_logprobs.py` — the public entry `compute_token_logprobs` at its production tile size. Vocabulary sizes are the real ones of mainstream models (LLaMA/LLaMA2/Mistral 32000, GPT-2 50257, ChatGLM 65024, LLaMA3 128256, Qwen2 151936) crossed with the `topk` values the sampler asks for (1, 2, 5, 10, 32, 64) and a batch size from the 1..64 request range. It also covers the `(1, 1)` edge case, extreme logits, determinism, row strides, and fully masked tiles/tails with FP32/FP16/BF16 inputs.
- `test_log_softmax.py` — the kernel invoked directly with `BLOCK_SIZE = 1024`, guarding the small-tile configuration.
- `test_compute_topk_logprobs.py` — `compute_topk_logprobs` end to end: token ids, logprobs and `selected_token_ranks` against `torch.topk` / `torch.log_softmax`.

The reference is `torch.nn.functional.log_softmax` in fp32 followed by `torch.gather`. As a reduction-type operator over the vocabulary, the tolerance is `atol=1e-4, rtol=1e-5` for fp32 input and `atol=1e-3, rtol=1e-4` for fp16 input.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_token_logprobs.py
```
