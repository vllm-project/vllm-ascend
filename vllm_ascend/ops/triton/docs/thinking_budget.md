# Thinking-budget compatibility kernels

## Overview

Model Runner V2 uses two upstream Triton functions while applying a per-request
thinking token budget:

- `_load_effective_token`, a helper called by `_thinking_budget_kernel` while
  scanning committed and in-flight token IDs.
- `_update_committed_marker_cache_kernel`, which incrementally tracks the most
  recent reasoning start and natural-end markers in committed token history.

The Ascend implementations live in
`vllm_ascend/ops/triton/sample/thinking_budget.py`. They are rebound onto the
upstream module by `vllm_ascend/patch/worker/patch_triton.py`; the
upstream `ThinkingBudgetState`, `apply_thinking_budget`, and
`_thinking_budget_kernel` remain unchanged.

## `_load_effective_token_ascend`

For an effective token position `pos`, the helper selects between two buffers:

```text
pos < total_len: all_token_ids[req_state_idx, pos]
otherwise:       input_ids[cur_req_first_pos + pos - total_len + 1]
```

The upstream helper expresses this selection with a runtime branch and returns
from both branches. When the helper is called inside a dynamic `tl.range`,
affected Triton-Ascend versions fail during TTIR-to-Linalg conversion while
materializing pointer arguments and the helper result. The Ascend helper issues
two complementary masked loads and selects the scalar with `tl.where`, giving
the compiler one explicit return path without changing the selected token.

This workaround can be deleted after the new Q4 Triton-Ascend release is
available. That release is expected to contain the fix for the helper-call
lowering issue.

## `_update_committed_marker_cache_kernel_ascend`

The marker-cache implementation keeps the upstream algorithm and data layout.
It rewrites three-term runtime conditions as nested
two-term conditions:

```text
A and B and C  ->  (A and B) and C
```

This applies to both cold-scan entry and its backward block loop. It avoids the
unsupported chained-boolean lowering in older Triton-Ascend versions.

This workaround can be deleted when Triton-Ascend 3.6.0 is the minimum
supported version.

## History-address range

History offsets are measured in int32 **elements**, not bytes. For a contiguous
`[max_num_seqs, max_model_len]` history, the largest valid offset is
`max_num_seqs * max_model_len - 1`. All offsets fit in signed int32 when this
product is at most `2**31`; in particular, `max_num_seqs <= 2048` together with
`max_model_len <= 1048576` is within that range. Widening the row index is not
needed for that bounded range.

The scheduler configuration does not impose a 2048 upper bound on
`max_num_seqs`. Request state allocates the configured capacity and assigns
free slots from the highest index first, so a large history offset does not
require that many concurrently active requests. Both compatibility kernels
widen the request index before multiplying by the row stride. This preserves
address arithmetic for configured histories whose reachable offsets exceed
signed int32, consistent with upstream staged history writes.

The address test uses `AscendRequestState` and normal request insertion with
scheduler-validated capacities `(3, 8)` and `(8193, 262144)`. The larger case
allocates a contiguous history of about 8 GiB, with ordinary row stride 262144;
its first request occupies slot 8192, whose row offset is `2**31`. It does not
construct an `as_strided` view. This is a production-state allocation boundary
test, not a claim that a complete model deployment or workload at that capacity
has been validated.

## Inputs and state

| Name | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `req_ids_ptr` | `[num_reqs]` | int32 | Active request-state indices. |
| `thinking_token_budget_ptr` | `[max_num_reqs]` | int32 | Per-request budget; a negative value disables the request. |
| `all_token_ids_ptr` | `[max_num_reqs, max_model_len]` | int32 | Committed token history. |
| `total_len_ptr` | `[max_num_reqs]` | int32 | Committed length per request. |
| `cached_last_start_ptr` | `[max_num_reqs]` | int32 | Latest reasoning-start position, or `-1`. |
| `cached_last_end_ptr` | `[max_num_reqs]` | int32 | Latest natural-end position, or `-1`. |
| `cached_scan_pos_ptr` | `[max_num_reqs]` | int32 | Earliest position needed by the next incremental scan. |
| `reasoning_start_token_ids_ptr` | `[START_LEN]` | int32 | Reasoning-start marker sequence. |
| `natural_reasoning_end_token_ids_ptr` | `[NATURAL_END_LEN]` | int32 | Natural reasoning-end marker sequence. |

The marker-cache kernel launches one Triton program per active request. The
helper is compiled as part of the upstream `_thinking_budget_kernel`, which
launches one program per logit row.

## Validation

The NPU test covers budget forcing, unmodified pre-budget logits, single- and
multi-token end markers, natural-end handling, incremental long-history scans,
plain requests, and oversized budgets:

```bash
pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_thinking_budget.py
```
