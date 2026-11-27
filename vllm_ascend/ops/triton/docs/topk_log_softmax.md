# topk_log_softmax

> Source: `vllm_ascend/worker/v2/sample/logprob.py` (`_topk_log_softmax_kernel`, exposed as `compute_token_logprobs`).

## Description

- **Function**: Computes the log-probability of a selected set of token ids per request. It fuses `log_softmax` over the whole vocabulary with the gather at the requested token positions, so the `[num_reqs, vocab_size]` intermediate that a `log_softmax` + `gather` pair would materialize is never written to global memory. It backs the model-runner-v2 sampler logprob path (`compute_token_logprobs`) and, through `compute_topk_scores`, the sampler / prompt-logprob / rejection-sampler logprob outputs.
- **Formula**: for logits row `i` and candidate slot `k`, with `t = token_ids[i, k]`:
    - `m_i = max_j logits[i, j]`
    - `lse_i = log(sum_j exp(logits[i, j] - m_i))`
    - `output[i, k] = logits[i, t] - lse_i - m_i`

  The max subtraction is the standard numerically stable form: the reduction is accumulated in fp32 even when `logits` is fp16/bf16, so `exp` never overflows for the value ranges produced by a language-model head.
- **Algorithm flow** (processed row by row, independently):
  1. Launch one program per logits row. A request can contribute multiple rows during speculative decoding. Each program reads its row at `row_ptr = logits_ptr + req_idx * logits_stride`.
  2. Pass 1 — running max: loop over the vocabulary in `BLOCK_SIZE` tiles, load each tile masked with `other=-inf`, and fold it into the running maximum with `tl.maximum(..., propagate_nan=tl.PropagateNan.ALL)` followed by `tl.max`. The result is cast to fp32.
  3. Pass 2 — sum of exponentials: loop over the same tiles again, cast to fp32, accumulate `sum(exp(logits - m_i))`, then take `lse_i = log(se)`.
  4. Gather: loop over candidates in blocks of at most 8192 columns. Use `candidate_offsets < topk` when loading token IDs and logits and storing the resulting fp32 logprobs.

  The row is read twice from global memory rather than staged in UB, which keeps UB usage bounded by `BLOCK_SIZE` instead of by `vocab_size`. `multibuffer=False` is passed at launch for the same reason.
- **Supported devices**: Atlas A2, Atlas A3, and 950PR/950DT. The large-candidate update was validated on A3 and 950DT in the MRV2 eager service. The direct kernel was also tested in NPU graph replay with changing candidate IDs and logits.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` (`logits_ptr`) | Input | Raw logits of the LM head, `[num_reqs, vocab_size]` | fp32 / fp16 / bf16 | ND |
| `token_ids` (`topk_ids_ptr`) | Input | Token ids whose logprob is requested, `[num_reqs, topk]`; cast to int64 by the wrapper | int32 / int64 | ND |
| `logits_stride` | Input (attribute) | Row stride of `logits`, i.e. `logits.stride(0)` | int32 | scalar |
| `topk_ids_stride` | Input (attribute) | Row stride of `token_ids`; may be zero for broadcast rows | int32 | scalar |
| `topk` | Input (attribute) | Number of token ids per request (`token_ids.shape[1]`) | int32 | scalar |
| `vocab_size` | Input (attribute) | Vocabulary size, i.e. `logits.shape[1]` | int32 | scalar |
| `BLOCK_SIZE` | Attribute (`tl.constexpr`) | Vocabulary tile length of both reduction loops; pinned to `12944` by `compute_token_logprobs` | int32 | scalar |
| `TOPK_BLOCK_SIZE` | Attribute (`tl.constexpr`) | Lane count of the gather, `min(max(next_power_of_2(topk), 2), 8192)` | int32 | scalar |
| `logprobs` (`output_ptr`) | Output | Log probabilities of the requested tokens, `[num_reqs, topk]` | fp32 | ND |

## Constraints

- `logits`: 2-D `[num_reqs, vocab_size]`, fp32 / fp16 / bf16, with a nonempty batch and unit column stride. Each row uses `logits_stride`. The tail of each vocabulary tile is masked with `-inf`, and one program handles each row.
- `token_ids`: `[num_reqs, topk]`, integer, values in `[0, vocab_size)`. The kernel performs no bounds check; an out-of-range id reads out of bounds. The last dimension must have unit stride. Rows use `topk_ids_stride`, including broadcast rows with stride zero; the output is contiguous.
- `topk >= 1`. `TOPK_BLOCK_SIZE` is `constexpr`, so its width is capped at 8192 to bound UB use for large candidate sets; `topk = 0` is handled by the caller (`compute_topk_scores` always concatenates the sampled token, giving at least one column).
- `logprobs` is always fp32, independent of the `logits` dtype; the reduction is accumulated in fp32 as well.
- `BLOCK_SIZE = 12944` is tuned against the UB budget rather than being a free tuning knob. Raising it, or reintroducing multi-buffering, risks a UB overflow; the `propagate_nan=tl.PropagateNan.ALL` form of `tl.maximum` in the first loop is likewise a compatibility workaround for the NaN handling introduced by the NPU-IR upgrade (see [#9193](https://github.com/vllm-project/vllm-ascend/pull/9193)) and must not be simplified back to a plain `tl.maximum`.
- `compute_token_logprobs` returns a new tensor and leaves both inputs unchanged. Both inputs must be on the same device.
- Inference only: there is no backward pass.

## Origin and Differences

- **Origin**: adapted from vLLM's GPU implementation `vllm/v1/worker/gpu/sample/logprob.py` (recorded in the file header). The Ascend version is installed over the upstream symbols by `vllm_ascend/patch/worker/patch_triton.py` (`logprob.compute_token_logprobs`, and `compute_topk_scores` in `sampler`, `prompt_logprob` and `rejection_sampler`).
- **Differences**:
    - Candidate gathers follow upstream [#46746](https://github.com/vllm-project/vllm/pull/46746) with bounded streaming; Ascend uses an 8192-column cap rather than the upstream 1024-column cap.
    - NPU adaptation for performance: the vocabulary tile is pinned to `BLOCK_SIZE = 12944` and the kernel is launched with `multibuffer=False`, both sized against the UB budget of the vector core; `tl.maximum` carries `propagate_nan=tl.PropagateNan.ALL` as a compatibility workaround after the NPU-IR upgrade. The companion `_ranks_kernel` used by `compute_topk_scores` is parallelized over `get_vectorcore_num()` cores with a `rows_per_core` row loop (and `do_not_specialize` on `batch_size` / `rows_per_core`) instead of one program per request, because an NPU has far fewer, much wider cores than a GPU has SMs.
    - Modified for a specific vllm-ascend logic or different input parameters: `compute_topk_scores` builds `logprob_token_ids` with `torch.topk` + `torch.cat` and then calls this kernel once for the concatenated ids. Custom candidate requests use the Ascend fill kernel and `expanded_idx_mapping` to preserve request-specific token order. Default requests in a mixed batch retain top-k behavior. Invalid columns are masked to negative infinity; logits mode gathers scores without normalization.

## JIT Specialization

The production wrapper uses a fixed vocabulary tile and chooses a power-of-two gather tile from 2 through 8192. Both tile widths are compile-time constants used by `tl.arange`. Candidate counts above 8192 share the same tile width. The total candidate count and vocabulary size remain runtime arguments, and the batch size determines the grid. Tensor dtype, alignment and runtime scalar specialization classes can still produce distinct compiled variants.

## Test Cases

Accuracy tests live under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`:

- `test_compute_token_logprobs.py` — the public entry `compute_token_logprobs`, i.e. the kernel at its production `BLOCK_SIZE = 12944`. Vocabulary sizes are the real ones of mainstream models (LLaMA/LLaMA2/Mistral 32000, GPT-2 50257, ChatGLM 65024, LLaMA3 128256, Qwen2 151936) crossed with the `topk` values the sampler asks for (1, 2, 5, 10, 32, 64) and a batch size from the 1..64 request range. It also covers the `(1, 1)` edge case, extreme logits rows (`+100`, `-100`, all-zero, all-one), run-to-run determinism, and fp16 input.
- `test_log_softmax.py` — the kernel invoked directly with `BLOCK_SIZE = 1024`, guarding the small-tile configuration.
- `test_compute_topk_scores.py` — `compute_topk_scores`: default/custom token IDs, logits/logprobs, metadata, actual score bindings and `selected_token_ranks`, with PyTorch references. Large-candidate cases cover 8192, 32768 and the complete 152064-token vocabulary, including mixed custom/default batches and both score modes.

The reference is `torch.nn.functional.log_softmax` in fp32 followed by `torch.gather`. As a reduction-type operator over the vocabulary, the tolerance is `atol=1e-4, rtol=1e-5` for fp32 input and `atol=1e-3, rtol=1e-4` for fp16 input.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_token_logprobs.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_log_softmax.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_scores.py
```
