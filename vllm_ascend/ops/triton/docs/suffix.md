# Suffix speculative decoding

## Description

- **Function**: Generate draft tokens from request history and an optional
  device-resident corpus of completed responses. This is a shared Model Runner
  V2 operator; it does not depend on a model architecture. The target model
  verifies its drafts.
- **Formula**: Local matches are ranked by
  `match_length * position_radix + end_position`. Corpus votes keep only
  candidates consistent with the chosen continuation prefix. A chain's score
  is its sum of prefix-support counts divided by the initial candidate count.
  Adaptive confidence is `(accepted + 1) / (observed + 2)` for valid slots
  and zero for filler slots.
- **Algorithm flow**: The upstream speculator calls `launch_suffix_observe`
  (adaptive only), `launch_suffix_local_scan`, and `launch_suffix_finalize`.
  A corpus match must be strictly longer to beat the local match.
  `launch_suffix_ingest` appends completed documents before their request
  slots are released. Finalize selects one draft per request; ingest and
  observe also update shared state.
- **Supported modes**: Kernel compilation and execution have been validated
  on Ascend A3 and A5/950. Eager mode and full decode graphs have been
  exercised on A3. Adaptive verification requires full decode graphs and an
  attention backend that supports variable query lengths.

## Parameters

`B` is the active batch size, `R` the request capacity, `L` the history
capacity, `K` the draft width, `N` the scan block count, `F` the document
count, and `C` the corpus capacity. Upstream may allocate buffers for the
full request capacity; shapes with `B` describe their active rows.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `position_radix`, `num_blocks`, `launch_options` | Attribute | Score radix, real scan block count and launch constants | Integers and booleans | Host scalars |
| `token_ids`, `total_len` | Input | Dense history and valid end offsets, indexed by request slot | int32 | `[R, L]`, `[R]` |
| `idx_mapping`, `num_sampled`, `num_rejected` | Input | Batch row to request slot and target verification counts | int32 | `[B]` each |
| `last_sampled` | Input | Last sampled tokens used to fill invalid draft slots | int64 | `[R]` |
| `scratch` | Output / Input | Best packed match per scan block; local scan writes it and finalize reads it | int64 | `[B, N]` |
| `drafts`, `num_valid` | Output | Fixed-width draft rows and actual retrieved token counts | int64, int32 | `[B, K]`, `[B]` |
| `corpus`, `table` | Input / Output | Token ring and two hash tables, read by finalize and updated in place by ingest | int32, int64 | `[C]`, `[2, NUM_BUCKETS, SLOTS]` |
| `docs`, `num_docs`, `corpus_len` | Input / Attribute | Finished request slot and history start per row, document count and corpus capacity | int32, host integers | `[F, 2]`, scalars |
| `head`, `doc_seq`, `head_out`, `doc_seq_out` | Input / Output | Ingest reads the ring cursor and document sequence and writes their next values to separate outputs | int64 | `[1]` each |
| `accept_stats` | Input / Output | Counters by source and match-length bucket, draft position, and accepted/observed field | int64 | `[18, K, 2]` |
| `last_bucket`, `last_valid` | Input / Output | Previous draft metadata by request slot; observe consumes the bucket marker | int32 | `[R]` each |
| `confidence` | Output | Smoothed acceptance estimate when adaptive verification is enabled | float32 | `[B, K]` |

## Constraints

- Tensors reside on one device. Histories and scratch use their actual row
  strides and require contiguous columns. Other buffers use the contiguous
  layouts above. Launch wrappers retain the upstream stride arguments for
  positional ABI compatibility. Active request counts come from
  `idx_mapping.shape[0]`, not the capacity of persistent buffers. The plain
  launch functions reconstruct their NPU grids without a grid argument.
- Draft width, block width and match depth are positive. Internal power-of-two
  padding does not add output slots. Corpus lookup needs a match depth of at
  least two. The two hash tables index 4-token and 2-token keys and keep
  eight candidate slots per bucket. When the corpus is disabled, the parent
  supplies placeholder buffers that the lookup path does not read. Token IDs
  are nonnegative; `-1` separates documents.
- Ingest uses the device history length and can include tokens beyond the
  response stop point. After ring wraparound, lookup keeps the newest
  matching entry for each position when both entries are present. It cannot
  recover the current document sequence when only a stale entry remains.
- The wrappers return `None` and write to caller-owned output and state
  buffers. Observe clears consumed bucket markers in a separate launch on the
  same stream. The parent copies the cursor and document sequence outputs
  back after ingest.
- Adaptive observe uses `next_power_of_2(B)` as a compile-time parameter.
  Crossing a boundary such as 64 to 65 can compile a new variant on a cold
  cache. Draft width, match depth and launch constants can also change
  variants. The module has no explicit JIT warmup registration.

## Origin and Differences

- **Origin**: [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
  Upstream owns allocation, proposal ordering and request lifecycle.
- **Differences**: The NPU implementation separates match state from indirect load masks, counts
  overlaps before committing ingest stores, and replaces observer atomics
  with per-bucket reductions followed by a marker-clear launch.
  Non-power-of-two draft widths use scalar statistics accesses.
- **A5 compatibility**: Continuations stop at the first document separator
  using a minimum-position reduction. This avoids incorrect results from the
  compare/cumsum/select path on narrow continuation rows with Triton-Ascend
  3.6.0.post2. Finalize pads its internal draft width to at least two to avoid
  a unit-width boolean gather compilation failure for `K=1`. Padding does
  not change the requested draft width or the returned tensor shape.

## Test Cases

The suite compares local scan and ingest to CPU references, exercises corpus
voting and wraparound, and checks observer counts and confidence. Integer
outputs require exact equality; floating-point confidence uses approximate
comparison.

The feature matrix includes `K=1` with corpus lookup enabled at match depth
two, with adaptive verification both enabled and disabled. Corpus voting,
document-boundary and short-history cases cover the narrow-row truncation
path. The 54-case suite passed on A3 and A5 with CANN 9.3.0 and
Triton-Ascend 3.6.0.post2, using vLLM `ba15685ab7`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_suffix_speculator.py
python -m pytest -q tests/ut/ops/triton/spec_decode/test_suffix_launchers.py
```

## Example

N/A for a standalone user call. The speculator invokes the launchers.
See the tests above for examples of caller-owned inputs and outputs.
