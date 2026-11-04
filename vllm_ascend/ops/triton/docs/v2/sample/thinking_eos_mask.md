# Thinking EOS mask kernels

## Description

- **Function**: while a request is inside its think block, masks the configured
  EOS token ids to `-inf` on every logits row of the request, so a mid-think
  EOS can no longer end generation with an empty `content` field (upstream
  vllm#55420).
- **Python entry point**: `apply_thinking_eos_mask` in
  `vllm_ascend/ops/triton/v2/sample/thinking_eos_mask.py`.
- **Triton kernels**: `_update_thinking_mask_cache_kernel`,
  `_thinking_eos_mask_kernel`.
- **Integration**: `vllm_ascend/patch/worker/patch_thinking_eos_mask.py`
  replaces the V2 sampler's `ThinkingBudgetState` with
  `ThinkingEosMaskV2State`, whose `apply()` calls the entry point after the
  stock kernelized budget path.

Enabled by `VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS` (comma-separated token
ids; unset keeps stock behavior); implicit think-end markers (e.g. the Qwen3
tool-call start token, for a tool call issued straight from thinking) come
from `VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS`.

A request is in think iff the last reasoning-start marker occurs after every
natural reasoning-end *or* implicit-end marker in the **committed** token
history:

```text
in_think = last_start >= 0 and last_start > last_end
last_end = max(last natural-end match, last implicit-end match)
```

In-flight draft tokens are deliberately excluded from the state: every row of
an in-think request is masked, which also covers the spec-decode verify and
bonus rows, so with MTP enabled EOS can neither be sampled nor
drafted-and-accepted mid-think. This matches the V1 holder
(`vllm_ascend/sample/thinking_eos_mask.py`).

## Data flow

`_update_thinking_mask_cache_kernel` launches one Triton program per active
request and mirrors the upstream thinking-budget marker cache: a cold scan
walks the committed history backward in 1024-token blocks, stopping at the
first block with a marker; afterwards an incremental scan only covers the
newly committed tail. A position counts as an end when it starts the natural
end sequence *or* matches any single-token implicit end marker (folded into
the same `last_end` via `max`). `cached_scan_pos` is rewound to
`total_len - (max(START_LEN, NATURAL_END_LEN) - 1)` so multi-token markers
that only complete in a later step are re-scanned; implicit ends are single
tokens and never extend the window.

`_thinking_eos_mask_kernel` launches one program per logits row, resolves the
row's request via `expanded_idx_mapping`, and stores `-inf` at every
configured EOS column when the cached marker positions say the request is in
think.

The kernelized path keeps the whole step on device: no per-request D2H copy
of `total_len` / `all_token_ids`, no CPU marker scan, and no per-step H2D of
`(row, token)` index pairs, which the earlier Python-side last-occurrence
scan paid on every sampling step.

## Parameters and state

| Name | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `idx_mapping` | `[num_reqs]` | int32 | Active request-state indices (cache-kernel grid). |
| `expanded_idx_mapping` | `[num_logits_rows]` | int32 | Row -> persistent request slot. |
| `all_token_ids` | `[max_num_reqs, max_model_len]` | int32 | Committed token history (prompt included). |
| `total_len` | `[max_num_reqs]` | int32 | Committed length per request. |
| `cached_last_start` | `[max_num_reqs]` | int32 | Latest reasoning-start position, or `-1`. |
| `cached_last_end` | `[max_num_reqs]` | int32 | Latest natural-or-implicit end position, or `-1`. |
| `cached_scan_pos` | `[max_num_reqs]` | int32 | Earliest position needed by the next incremental scan. |
| `reasoning_start_token_ids` | `[START_LEN]` | int32 | Reasoning-start marker sequence. |
| `natural_reasoning_end_token_ids` | `[NATURAL_END_LEN]` | int32 | Natural reasoning-end marker sequence. |
| `implicit_end_token_ids` | `[IMPLICIT_END_LEN]` | int32 | Single-token implicit end markers; may be empty. |
| `eos_mask_token_ids` | `[EOS_LEN]` | int32 | EOS token ids masked to `-inf` in think. |
| `logits` | `[num_logits_rows, vocab_size]` | float32 | Modified in place. |

The cache tensors are owned by `ThinkingEosMaskV2State`, separate from the
budget path's, because the mask tracks every request rather than only the
requests that set a `thinking_token_budget`. `apply_staged_writes()` resets
them (`-1`/`-1`/`0`) for every (re)used request slot.

## Constraints

- The state requires a reasoning config with non-empty start and natural end
  markers; otherwise the mask stays disabled and behavior is stock.
- The budget-exhaustion forcing kernels launch before the mask kernels, so
  when both configure the same token id the mask wins - matching the V1
  holder's ordering (parent forcing first, then the EOS mask).
- Interaction with later sampling stages is safe: temperature scaling and
  min_p/top-k/top-p preserve `-inf`.
- Runtime conditions follow the nested two-term form
  (`(A and B) and C`) required by older Triton-Ascend versions, matching
  `_update_committed_marker_cache_kernel_ascend`.

## Test cases

The NPU test covers the mask on plain decode rows, on spec-decode verify and
bonus rows, natural and implicit end handling, slot-recycle cold scans,
incremental long-history scans, coexistence with budget forcing, and the
unset-env stock behavior:

```bash
pytest -sv tests/e2e/pull_request/one_card/test_thinking_eos_mask.py
```
