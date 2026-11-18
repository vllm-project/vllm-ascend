# Step5 build_prefill_groups (CSA prefill grouping metadata)

## Description

- **This will be removed once triton-ascend is compatible with upstream implementations**
- **Function**: Computes, entirely on device, the prefill summary-update
  grouping metadata: which rows of a prefill chunk belong to one eight-token
  region of one request, which of those rows are the group's first (carry
  offset), and which summary region a complete group closes. Its output
  `[rows, 10]` int64 tensor (`[region, first, rows[8]]`) is consumed by
  `update_prefill`. Replaces a host-side grouping plan that would have
  needed pinned staging and a D2H sync; on Ascend it is "one synchronous
  launch, no host round trip". Called from
  `AscendStep5DSAMetadata._update_summary_state` before `update_prefill`.
- **Location**: `vllm_ascend/ops/triton/step5/summary_update.py` — kernel
  `_build_prefill_groups`, wrapper `build_prefill_groups`.
- **Formula** (per token row `r` of request `q = owners[r]`, span
  `[start, end)` from `query_start_loc`):
    - `head = (r == start) | (positions[r] % 8 == 0)` — a row heads a group
      iff it starts the request's chunk or lands on a region boundary.
    - Candidate rows: `rows = r + (offset - positions[r] % 8)` for
      `offset in [0, 8)`, kept when `head & (offset >= positions[r] % 8) &
      (rows < end)` (same request, same region, inside the chunk).
    - `valid = in_group & (slots[rows] >= 0)`;
      `first = min(valid ? offset : 8)` (the group's first valid row offset,
      `8` when none);
      `region = max(valid ? slots[rows] // 8 : -1)` — the summary region id
      the group's rows share (a token-unit slot maps to region `slot // 8`).
    - Stores `metadata[r] = [first < 8 ? region : -1, first,
      valid ? rows : -1 (x8)]`. Non-head rows write `region = -1` (their
      group is reported by the head row).
- **Algorithm flow** (one program per token row; integer logic only):
  1. Load the row's owner, request span, position and head flag.
  2. Gather the (up to) 8 candidate rows' slots with an in-group mask,
     `-1` for out-of-group lanes.
  3. Reduce `first` (min) and `region` (max) over the valid lanes and store
     the 10-int64 metadata row.
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend; execution
  evidence on Ascend950DT). Integer-only kernel, no graph capture on the eager
  prefill path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `slots` | Input | Summary-cache slot per row (token units), `-1` = padding | int64 | ND (1-D) |
| `positions` | Input | Absolute position of each row | int64/int32 | ND (1-D); stride passed |
| `owners` | Input | Request index per row | int32/int64 | ND (1-D) |
| `query_start_loc` | Input | `[num_requests + 1]` cumulative chunk row spans | int32/int64 | ND (1-D) |
| return `metadata` | Output | `[rows, 10]` grouping plan: `[region (-1 = no close), first offset, row indices rows[8] (-1 = absent)]` | int64 | ND 2-D, freshly allocated |

## Constraints

- Rows of one request must be contiguous in `[start, end)` and carry
  ascending positions (prefill chunk contract); the kernel never reads
  positions other than `positions[r]` for the head test — candidate rows are
  derived by position arithmetic.
- `owners[r]` must index `query_start_loc` validly (`<= num_requests - 1`).
- A group closes a region only if its eighth row (metadata slot 9) is a
  valid row — `update_prefill` enforces the rest.
- `rows == 0` returns an empty `[0, 10]` tensor without launching.
- Specialization: only `POSITION_STRIDE` (fixed, contiguous positions) is a
  constexpr — a single serving variant.
- Determinism: pure integer gather/reduce, output is exact (compared
  bit-exact in tests); no atomics, no cross-program writes.

## Origin and Differences

- **Origin**: Carried over from the upstream Triton implementation
  `vllm.models.step5.nvidia.ops.summary_update._build_prefill_groups` /
  `build_prefill_groups` (identical body and output contract).
- **Differences**: none in semantics; on Ascend the value is the call
  context — the metadata builder calls it directly instead of staging a
  host-computed plan through pinned buffers (the Ascend prefill plan is
  computed synchronously in the builder, with no cross-step event
  dependencies).

## Test Cases

Tests feed random prefill batches (unaligned region starts, partial groups,
`slot = -1` rows, multi-request spans, empty batch) and compare the
`[rows, 10]` int64 metadata exactly against an independent pure-torch
implementation of the grouping rules (integer output → exact comparison),
then feed the metadata to `update_prefill` end-to-end.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_summary_update.py -k groups
```
