# topk_log_softmax

See `vllm_ascend/worker/v2/sample/logprob.py` for `compute_token_logprobs` and `compute_topk_logprobs`.

## Description

- **Function**: Computes the log-probability of a selected set of token ids per request. It fuses `log_softmax` over the whole vocabulary with the gather at the requested token positions, so the `[num_reqs, vocab_size]` intermediate that a `log_softmax` + `gather` pair would materialize is never written to global memory. It backs the model-runner-v2 sampler logprob path (`compute_token_logprobs`) and, through `compute_topk_logprobs`, the sampler / prompt-logprob / rejection-sampler logprob outputs.
- **Formula**: for request `i` and slot `k`, with `t = token_ids[i, k]`:
    - `m_i = max_j logits[i, j]`
    - `lse_i = log(sum_j exp(logits[i, j] - m_i))`
    - `output[i, k] = logits[i, t] - lse_i - m_i`

  The max subtraction is the standard numerically stable form: the reduction is accumulated in fp32 even when `logits` is fp16/bf16, so `exp` never overflows for the value ranges produced by a language-model head.
- **Row-wise fallback** for vocabularies below the online reduction threshold.
  1. Grid is `(num_reqs,)`: one program per request, each owning one full logits row (`row_ptr = logits_ptr + req_idx * logits_stride`).
  2. Pass 1 — running max: loop over the vocabulary in `BLOCK_SIZE` tiles, load each tile masked with `other=-inf`, and fold it into the running maximum with `tl.maximum(..., propagate_nan=tl.PropagateNan.ALL)` followed by `tl.max`. The result is cast to fp32.
  3. Pass 2 — sum of exponentials: loop over the same tiles again, cast to fp32, accumulate `sum(exp(logits - m_i))`, then take `lse_i = log(se)`.
  4. Gather: load `PADDED_TOPK` token ids for the request under the mask `k_offset < topk`, gather `logits[i, t]` at those ids, compute `logits - lse_i - m_i` in fp32 and store the `topk` valid lanes.

  The row is read twice from global memory rather than staged in UB, which keeps UB usage bounded by `BLOCK_SIZE` instead of by `vocab_size`. `multibuffer=False` is passed at launch for the same reason.
- **Online reduction** from 65,536 entries when ranks are requested and from 128,256 entries for token logprobs alone. The token-only path uses a higher threshold because it already avoids the separate rank launch.
  1. A program per request reads the vocabulary once in 12,288-column tiles. For each tile, it updates the running maximum, rescales the previous exponential sum, and adds the tile's contribution. A fully masked prefix contributes zero.
  2. For top-k results, the same tile load supplies the selected-token rank comparison. Integer rank counts accumulate in the program. The final rank includes the padding used in the upstream 8192-column reduction when the selected logit is `-inf`.
  3. The program gathers and normalizes the requested logits using the final maximum and exponential sum. It writes the rank when requested.

  The online path uses one launch for both operations and needs no intermediate buffer in global memory. The tile width and whether rank counts are requested are compile-time choices.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT Products. Used by vLLM model-runner-v2 sampling; it is installed over the upstream implementation in `vllm_ascend/patch/worker/patch_triton.py`. It runs in the sampler post-processing path, i.e. eagerly, outside ACL Graph capture.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` (`logits_ptr`) | Input | Raw logits of the LM head, `[num_reqs, vocab_size]` | fp32 / fp16 / bf16 | ND |
| `token_ids` (`topk_ids_ptr`) | Input | Token ids whose logprob is requested, `[num_reqs, topk]`; cast to int64 by the wrapper | int32 / int64 | ND |
| `logits_stride` | Input (attribute) | Row stride of `logits`, i.e. `logits.stride(0)` | int32 | scalar |
| `topk` | Input (attribute) | Number of token ids per request (`token_ids.shape[1]`) | int32 | scalar |
| `vocab_size` | Input (attribute) | Vocabulary size, i.e. `logits.shape[1]` | int32 | scalar |
| `BLOCK_SIZE` | Attribute (`tl.constexpr`) | Vocabulary tile length, 12944 for the row-wise fallback and 12288 for the online reduction | int32 | scalar |
| `PADDED_TOPK` | Attribute (`tl.constexpr`) | Lane count of the gather, `max(next_power_of_2(topk), 2)` | int32 | scalar |
| `logprobs` (`output_ptr`) | Output | Log probabilities of the requested tokens, `[num_reqs, topk]` | fp32 | ND |

## Constraints

- `logits` has shape `[num_reqs, vocab_size]` and dtype fp32, fp16, or bf16. The vocabulary dimension has unit stride and needs no alignment. The kernel masks the tail and uses `logits_stride` to locate each row.
- `token_ids` has shape `[num_reqs, topk]` and values in `[0, vocab_size)`. Rows may be padded or broadcast. Columns use unit stride. The kernels rely on callers to provide in-range IDs.
- Empty batches and empty token selections return empty outputs. Each power-of-two gather width can create a separate `PADDED_TOPK` specialization. With the online path, requests for ranks compile a separate variant.
- `logprobs` is always fp32, independent of the `logits` dtype; the reduction is accumulated in fp32 as well.
- `BLOCK_SIZE = 12944` is tuned against the UB budget rather than being a free tuning knob. Raising it, or reintroducing multi-buffering, risks a UB overflow; the `propagate_nan=tl.PropagateNan.ALL` form of `tl.maximum` in the first loop is likewise a compatibility workaround for the NaN handling introduced by the NPU-IR upgrade (see [#9193](https://github.com/vllm-project/vllm-ascend/pull/9193)) and must not be simplified back to a plain `tl.maximum`.
- Sampling and logprob computation run eagerly after the captured model forward. Batch size can vary across calls.
- Inference only: there is no backward pass.

## Origin and Differences

- The Ascend logprob kernels derive from vLLM's GPU implementation in `vllm/v1/worker/gpu/sample/logprob.py`. The existing worker patch routes sampler, prompt-logprob, and rejection-sampler calls to the Ascend implementation under the entry point exposed by each vLLM version. On versions with `compute_topk_scores`, standard logprobs use the Ascend rank and normalization kernels, while raw logits and custom token-ID handling use the upstream score entry point. The adapter can be removed when upstream exposes a backend-specific dispatch for the standard logprob kernels.
- **Differences**:
    - NPU adaptation for performance: the vocabulary tile is pinned to `BLOCK_SIZE = 12944` and the kernel is launched with `multibuffer=False`, both sized against the UB budget of the vector core; `tl.maximum` carries `propagate_nan=tl.PropagateNan.ALL` as a compatibility workaround after the NPU-IR upgrade. The companion `_ranks_kernel` used by `compute_topk_logprobs` is parallelized over `get_vectorcore_num()` cores with a `rows_per_core` row loop (and `do_not_specialize` on `batch_size` / `rows_per_core`) instead of one program per request, because an NPU has far fewer, much wider cores than a GPU has SMs.
    - Modified for a specific vllm-ascend logic or different input parameters: `compute_topk_logprobs` builds `logprob_token_ids` with `torch.topk` + `torch.cat` and then calls this kernel once for the concatenated ids. The upstream `logprob_token_ids_state` / `expanded_idx_mapping` / `max_per_req_token_ids` path is accepted for signature compatibility but not implemented — a non-zero `max_per_req_token_ids` logs a warn-once and falls back to the default token ids.

## Test Cases

Accuracy tests live under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`:

- `test_compute_token_logprobs.py` exercises the public wrapper with common model vocabulary sizes, multiple batch and gather widths, extreme logits, repeated runs, and fp16 input.
- `test_log_softmax.py` — the kernel invoked directly with `BLOCK_SIZE = 1024`, guarding the small-tile configuration.
- `test_compute_topk_logprobs.py` — `compute_topk_logprobs` end to end: token ids, logprobs and `selected_token_ranks` against `torch.topk` / `torch.log_softmax`.

The reference is `torch.nn.functional.log_softmax` in fp32 followed by `torch.gather`. As a reduction-type operator over the vocabulary, the tolerance is `atol=1e-4, rtol=1e-5` for fp32 input and `atol=1e-3, rtol=1e-4` for fp16 input.

The online-path regression and comparison benchmark use an absolute tolerance of `1e-5` and exact rank matches. Supply the baseline from the same checkout and collect a profiler trace for each version.

```bash
test_dir=tests/e2e/nightly/single_node/ops/singlecard_ops/triton
pytest -sv "$test_dir/test_compute_token_logprobs.py" "$test_dir/test_compute_topk_logprobs.py"
git show "$BASE_SHA:vllm_ascend/worker/v2/sample/logprob.py" > /tmp/logprob_baseline.py
python benchmarks/scripts/benchmark_mrv2_logprobs.py \
  --baseline /tmp/logprob_baseline.py --profile-dir ./logprob-profiler \
  | tee logprob-benchmark.jsonl
```

Run the streaming decode comparison from the baseline and candidate environments with the same model weights and runtime. Use the same MRV2 score binding in both environments so the comparison measures the kernel change. The script records first-token latency, time per output token, decode throughput, and the generated token IDs for greedy and seeded sampling.

```bash
# Baseline environment
VLLM_USE_V2_MODEL_RUNNER=1 python benchmarks/scripts/benchmark_mrv2_logprobs_e2e.py \
  --model "$MODEL_PATH" --output "$RESULT_DIR/e2e-baseline.json"

# Candidate environment
VLLM_USE_V2_MODEL_RUNNER=1 python benchmarks/scripts/benchmark_mrv2_logprobs_e2e.py \
  --model "$MODEL_PATH" --output "$RESULT_DIR/e2e-candidate.json" \
  --reference "$RESULT_DIR/e2e-baseline.json"
```
