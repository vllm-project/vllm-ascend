# _min_p_kernel

Source: `vllm_ascend/worker/v2/sample/min_p.py` (host wrapper: `apply_min_p`).

## Description

- **Function**: Applies min-p sampling filtering to the sampler logits **in place**. For every token row, tokens whose logit is below `max_logit + log(min_p)` (equivalently, whose probability is below `min_p * max_prob`) are masked out with `-inf`; rows whose request has `min_p == 0.0` are left untouched. It replaces `vllm.v1.worker.gpu.sample.states.apply_min_p` through `vllm_ascend/patch/worker/patch_v2/patch_triton.py`.
- **Formula**: For token row `t` with request index `r = expanded_idx_mapping[t]` and `p = min_p[r]`:

    ```text
    if p == 0.0:
        out[t, :] = in[t, :]                        # min-p disabled for this request
    else:
        threshold = max(in[t, :]) + log(p)          # computed in fp32
        out[t, v] = -inf            if in[t, v] < threshold
        out[t, v] = in[t, v]        otherwise
    ```

    In the softmax domain this is `prob[t, v] < p * max(prob[t, :])  ->  masked`, since `log` is monotonic and the row max cancels out.
- **Algorithm flow** (processed row by row, independently):
  1. Host side (`apply_min_p`): read `num_tokens, vocab_size` from `logits`, build a persistent grid of `core_nums = min(num_tokens, get_vectorcore_num())` programs, choose the vocabulary tile `BLOCK_SIZE = min(next_power_of_2(vocab_size), 8192)`, and launch the kernel with `logits` passed as **both** the input and the output pointer (in-place update), with `multibuffer=False`.
  2. Kernel side: each program owns a contiguous slice of token rows — `tokens_per_block = cdiv(num_tokens, num_programs)`, `start_token = pid * tokens_per_block`, `end_token = min(start_token + tokens_per_block, num_tokens)` — and iterates over it, so any `num_tokens` is covered by the fixed grid.
  3. Kernel side, per row: load `req_state_idx = expanded_idx_mapping[token_idx]`, then `min_p = min_p[req_state_idx]` cast to fp32. If `min_p == 0.0`, the row is skipped entirely (no load, no store), which keeps mixed batches cheap.
  4. Kernel side, pass 1 (reduction): scan the vocabulary in `BLOCK_SIZE` tiles, masking the tail with `other=-inf`, and reduce to the row maximum `max_val`; cast it to fp32. Tiling means the whole vocabulary row never has to fit into UB.
  5. Kernel side, pass 2 (masking): compute `threshold = max_val + log(min_p)`, re-scan the vocabulary in `BLOCK_SIZE` tiles, apply `tl.where(logits < threshold, -inf, logits)` and store the tile back to the output row with the same tail mask.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the V2 worker sampler (`vllm_ascend/patch/worker/patch_v2/patch_triton.py`) in both the normal decode path and the speculative-decoding sampling path.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `in_logits_ptr` | Input | Sampler logits `[num_tokens, vocab_size]`; the wrapper passes `logits` here | fp32 | ND |
| `out_logits_ptr` | Output | Filtered logits `[num_tokens, vocab_size]`; the wrapper passes the same `logits` tensor, so the update is in place | fp32 | ND |
| `logits_stride` | Input (attribute) | Row stride of the logits tensor (`logits.stride(0)`) | int32 | scalar |
| `expanded_idx_mapping_ptr` | Input | Per-token mapping from token row to request state index `[num_tokens]` (a token row of a request reuses that request's `min_p`) | int32 | ND |
| `min_p_ptr` | Input | Per-request min-p values `[max_num_reqs]`, indexed by `expanded_idx_mapping`; `0.0` disables filtering for that request | fp32 | ND |
| `vocab_size` | Input (attribute) | Vocabulary size, i.e. the number of columns of the logits | int32 | scalar |
| `num_tokens` | Input (attribute) | Number of token rows; not specialized (`do_not_specialize`) so a changing batch size does not trigger recompilation | int32 | scalar |
| `BLOCK_SIZE` | Input (attribute) | Vocabulary tile size, compile-time `constexpr`, set to `min(next_power_of_2(vocab_size), 8192)` by the wrapper | int32 | scalar |

## Constraints

- `logits` must be 2-D `[num_tokens, vocab_size]` and fp32 (the sampler always produces fp32 logits); rows are addressed through `logits_stride`, so only the last dimension has to be contiguous.
- The update is in place: `apply_min_p` returns `None` and the caller's `logits` tensor is overwritten. Rows with `min_p == 0.0` are not written at all, which is safe only because the input and output pointers are the same tensor — passing two different tensors would leave those rows uninitialized in the output.
- `expanded_idx_mapping` has one entry per token row, with values in `[0, min_p.shape[0])`; out-of-range values would read past the `min_p` buffer. Both `expanded_idx_mapping` and `min_p` must be contiguous one-dimensional tensors because the kernel indexes them as `ptr + index` and receives no stride for either input.
- `min_p` values must lie in `[0.0, 1.0]`. `0.0` means "disabled"; a value of exactly `1.0` keeps only the row maximum. Negative values are invalid (`log(min_p)` would be `NaN`).
- The row maximum and the threshold are computed in fp32 regardless of the tile size, so results do not depend on `BLOCK_SIZE`.
- `BLOCK_SIZE` is capped at `8192` so that one `[BLOCK_SIZE]` fp32 tile (32KB) stays within UB even for very large vocabularies (e.g. `151936` for Qwen-class models); the two-pass loop then handles the rest of the row.
- `num_tokens` is dynamic: the grid is capped by the vector-core count and each program loops over its own token slice, so no recompilation and no host synchronization are needed. Graph mode: supported — the launch grid depends only on the device vector-core count and host-side shapes.
- The kernel is launched with `multibuffer=False`; the two vocabulary passes read the same row twice, so multi-buffering brings no benefit here.

## Origin and Differences

- **Origin**: Adapted from `vllm/v1/worker/gpu/sample/min_p.py` of vLLM (see the file header), which implements min-p sampling for the V1/V2 GPU sampler. It is installed over the upstream `states.apply_min_p` by the vllm-ascend worker patch.
- **Differences**:
    - NPU adaptation for performance: replaces the upstream one-program-per-token grid (`grid = (num_tokens,)`) with a persistent grid of `min(num_tokens, get_vectorcore_num())` programs, each looping over a contiguous slice of `tokens_per_block` rows — this keeps the number of programs matched to the vector cores and amortizes launch overhead for large batches; `BLOCK_SIZE` is raised from the fixed upstream `1024` to `min(next_power_of_2(vocab_size), 8192)` to cut the number of vocabulary passes while staying inside UB; the launch also sets `multibuffer=False`, an Ascend Triton launch option;
    - Modified for a specific vllm-ascend logic or different input parameters: the kernel takes separate `in_logits_ptr` / `out_logits_ptr` pointers instead of a single in/out pointer (the wrapper passes the same tensor for both, so behaviour is unchanged, but the kernel can be reused for an out-of-place variant), `num_tokens` is passed as a runtime argument marked `do_not_specialize` to avoid recompiling for every batch size, and the early exit for `min_p == 0.0` becomes a skipped loop iteration instead of an early `return` (a program now owns several rows).

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py` compares the kernel against a per-row PyTorch reference. Shapes use the vocabulary sizes of the models served on NPU — `(num_reqs, vocab_size)` of `(48, 102400)`, `(96, 102400)`, `(24, 151936)` (Qwen-class) and `(1, 32000)` (single-request decode) — with a reversed `expanded_idx_mapping` so the token-to-request mapping is non-identity, and `min_p` drawn from `[0.01, 0.5]`. Verification is two-fold, following the operator type: the `-inf` mask positions must match **exactly** (`torch.equal` on the `isinf` masks, since the masking decision is a comparison), while the surviving logits use the fp32 tolerance `rtol=1e-4, atol=1e-4`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_min_p.py
```
