# prepare_nonempty_keypool_inputs

## Description

- **Function**: Prepare the six inputs needed to keep the CANN pooled key
  indexer supplied with nonempty work. Append a zero query and weight row,
  append a sentinel request with one pooled key, clamp empty real key histories
  to one physical pool, and append a zero block-table row. The CANN operator
  still computes pooled-key scores and top-k indices.
- **Formula**: For `T` query rows, `R` requests, and concatenation `cat`:
  `Q' = cat(Q, 0)`, `W' = cat(W, 0)`,
  `E' = cat(E, T + 1)`, `L' = cat(max(L, 1), 1)`,
  `S' = cat(S, 0)`, and `B' = cat(B, 0)` along the first dimension.
- **Algorithm flow**: Normalize strided views with `contiguous()`, allocate
  independent outputs, and launch one Triton kernel to copy data and initialize
  the added rows. Wide tiles copy query, weight, and block-table elements;
  separate 16-element tiles handle int64 metadata. Contiguous inputs need no
  preliminary copies. The caller invokes CANN, removes the sentinel query
  result, and packs indices using the original positions and request boundaries
  to discard synthetic history and graph-padding candidates.
- **Supported modes**: Validated on Atlas A3 with BF16 query and weight tensors,
  in eager execution and NPU graph capture/replay. Other hardware is not
  validated by these results.

## Parameters

All inputs are required. The return value is a tuple of six tensors in the
order listed below. Outputs retain their corresponding input device and dtype.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query` | Input | `T` query rows, `H` heads, head dimension `D` | bfloat16 | `[T, H, D]`, TND |
| `weights` | Input | Per-query head weights | bfloat16 | `[T, H]` |
| `ends` | Input | Cumulative local query ends for `R` requests | int64 | `[R]` |
| `lengths` | Input | Number of completed pooled keys per request | int64 | `[R]` |
| `tails` | Input | Number of unpooled tail tokens per request | int64 | `[R]` |
| `table` | Input | Physical pooled-key cache block IDs; `C` columns | int32 | `[M, C]` |
| `padded_query` | Output | Original queries followed by one zero row | bfloat16 | `[T + 1, H, D]`, contiguous |
| `padded_weights` | Output | Original weights followed by one zero row | bfloat16 | `[T + 1, H]`, contiguous |
| `padded_ends` | Output | Original ends followed by `T + 1` | int64 | `[R + 1]`, contiguous |
| `padded_lengths` | Output | Real lengths clamped to at least one; sentinel length one | int64 | `[R + 1]`, contiguous |
| `padded_tails` | Output | Original tails followed by zero | int64 | `[R + 1]`, contiguous |
| `padded_table` | Output | Original table followed by one zero row | int32 | `[M + 1, C]`, contiguous |

## Constraints

- All inputs must be on the same NPU. Query/weight shapes and the three
  request-metadata lengths must agree. Inputs may be strided; outputs are
  contiguous and do not alias the inputs or other invocations.
- The GLM KPool integration uses `H = 32`, `D = 128`, and pool size four.
  `ends` is nondecreasing and lies in `[0, T]`; repeated ends represent empty
  local requests. Rows after the final end are graph padding. `lengths` is
  nonnegative and `tails` lies in `[0, 3]` for pool size four.
- `M >= R` and `C >= 1`. When `M > R`, unused table rows, including row `R`
  used by the sentinel request, must already contain valid block IDs. The
  appended row is zero-filled. Physical cache block zero must be allocated and
  readable; each real request's first block must also be valid when its history
  is empty. This helper does not allocate or initialize cache blocks.
- The caller uses causal CANN `TND`/`PA_BBND` indexing (`mask_mode = 3`). Clamping
  empty histories is safe only together with index packing that filters using
  the original metadata. Passing the synthetic candidates directly to attention
  would change the real request's semantics.
- Graph replay keeps tensor shapes and addresses fixed while metadata values
  may change, including all-empty local requests. Outputs are allocated per
  invocation and captured allocations are reused on replay. There is no
  device-to-host synchronization or data-dependent Python branch.

## Origin and Differences

- **Origin**: Developed for vLLM Ascend's CANN KPool integration. This input
  preparation helper is not a replacement implementation of the CANN indexer.
- **Differences**: Fuse concatenation, zero initialization, and length clamping
  into one kernel for contiguous inputs. A sentinel provides effective work
  when a DSA-CP shard has no completed pooled keys, avoiding the A3 CANN
  empty-work output-initialization path. Small metadata tiles avoid expensive
  wide int64 vector lowering on A3. No CANN package files are modified.

## Test Cases

The accuracy suite uses the GLM head count, head dimension, and pool size above.
It compares all six prepared tensors against an independent PyTorch reference
with exact equality (`rtol = 0`, `atol = 0`), checks contiguous and strided
inputs, and verifies that inputs remain unchanged. Shape variations exercise
single-row inputs, graph padding, metadata tile boundaries, and wide tables.

CANN integration cases cover empty local histories, zero-query requests,
fully padded replay, and transitions to nonempty histories without recapture.
Packed indices are compared with the expected real-token indices. A nonempty
top-k case compares with an unadapted CANN call. These are integer comparisons;
there is no floating-point tolerance to relax. Run with a compatible CANN
pooled-indexer custom-op package installed.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_nonempty_keypool_inputs.py
```
