# Step5 update_prefill (CSA region summary, prefill)

## Description

- **This will be removed once triton-ascend is compatible with upstream implementations**
- **Function**: Folds a prefill chunk's indexer rows into the CSA state in
  two launches: `_update_prefill` closes every region whose eighth row is in
  this chunk (replaying the region's earlier rows — whether stored by an
  earlier step in the ring or by this chunk — as one flat masked tile), then
  `_write_ring` stores every open-region row of the chunk into the ring.
  Consumes the `[rows, 10]` int64 grouping metadata produced by
  `build_prefill_groups`. Called from
  `AscendStep5DSAMetadata._update_summary_state` on batches with prefill
  rows (after the decode/verify rows of a mixed batch).
- **Location**: `vllm_ascend/ops/triton/step5/summary_update.py` — kernels
  `_update_prefill` and `_write_ring`, wrapper `update_prefill` (helpers
  `_store_summary`, `summary_layout`).
- **Formula**:
    - Close (per group metadata row with `region >= 0` and `rows[7] >= 0`,
      i.e. the group holds the region's eighth row):
      let `first` = offset of the group's first row, `head` = `rows[first]`,
      `position` = `positions[head]`, `base` = `ring_base[head]`.
      Carry tile: ring rows at positions `[position - first, position)`
      masked by `offset < first`; chunk tile: the group's up-to-8 rows
      masked by `row >= 0`. In fp32:
      `m = max(carry z, chunk z)`, `w = exp(z - m)`,
      `num = Σ_carry w*k + Σ_chunk w*k`, `den = Σ_carry w + Σ_chunk w`,
      `summary = num / max(den, 1e-20)` stored
      `fp32 -> bf16 -> fp8 e4m3` at token-slot `region * 8`.
      (Invalid chunk rows load `z = -inf`, so their weight is
      `exp(-inf) = 0`; the region always closes here, so `m` is finite.)
    - Ring store (per row with `ring_slots[row] >= 0`):
      `Ring[slot * 512 + dim] = key`, `Ring[slot * 512 + 256 + dim] = z`.
- **Algorithm flow**:
    1. `_update_prefill`, grid `(rows,)`: only group-head metadata rows with
       a complete region close it; every ring read of the step happens in
       this launch.
    2. `_write_ring`, grid `(rows,)`: stores the chunk's open-region rows.
       Launch order guarantees a long chunk cannot overwrite ring rows that
       closes in launch 1 still have to read.
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend; execution
  evidence on Ascend950DT). Prefill (and mixed) batches run eager on this backend, so
  the wrapper allocates nothing capturable-dependent; both launches are
  shape-driven.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key` | Input | Index K rows of the chunk, `[rows, groups, 256]` | bf16 | ND; `stride(0)`/`stride(2)` passed to the kernels |
| `z` | Input | Index Z rows, same layout | bf16 | ND |
| `metadata` | Input | Grouping from `build_prefill_groups`, `[rows, 10]`: `[region, first, rows[8]]`, non-head rows `-1` | int64 | ND 2-D |
| `positions` | Input | Absolute position per row | int64 | ND (1-D); stride passed |
| `ring_base` | Input | First ring row of the row's request block, `-1` skips | int64 | ND (1-D) |
| `ring_slots` | Input | Ring destination per row (`base + pos % capacity`), `-1` skips the store | int64 | ND (1-D) |
| `mean` | Output | Summary cache `[blocks, block_size // 8, 256]` | uint8 (fp8 e4m3 payload) | ND |
| `ring` | Output | Tail-ring rows, flat `[total_rows * 512]` | bf16 | 1-D flat |
| `capacity` | Attribute | Ring rows per request block (constexpr) | int | - |
| `MEAN_BLOCK`, `BLOCK_SIZE` | Attribute | From `summary_layout(mean)` (constexpr) | int | - |

## Constraints

- `metadata` must come from `build_prefill_groups` over the same row set
  (the kernel trusts the `[region, first, rows[8]]` contract).
- A region closes only when its eighth row is in this chunk
  (`metadata[row, 9] >= 0`); regions whose remaining rows arrive in later
  chunks are left open with their rows in the ring.
- The carry replay assumes ring rows `[position - first, position)` were
  stored by an earlier step; `_write_ring` of THIS call runs after all
  closes, preserving that assumption.
- `ring_slots[row] < 0` rows (padding, or rows outside the ring's keep
  window) are not stored. Valid ring slots must be unique within a launch —
  the ring metadata's keep rule (`pos >= seq_len - capacity`) guarantees
  this, so a chunk longer than the ring capacity never wraps onto its own
  rows.
- Kernel launches are skipped when `metadata.shape[0] == 0` (the
  `_write_ring` grid still launches over `ring_slots` rows).
- Specialization: all strides/geometry are constexpr and fixed per
  deployment (contiguous inputs, fixed capacity/cache layout) — one serving
  variant; `enable_fp_fusion=False` on the close launch.
- In-place behavior: `ring` and `mean` are mutated in place; the close
  launch never stores ring rows (the closing row's ring store is skipped by
  the metadata contract — its content is folded into the summary).

## Origin and Differences

- **Origin**: Adapted from the upstream Triton implementation
  `vllm.models.step5.nvidia.ops.summary_update._update_prefill` /
  `_write_ring` / `update_prefill` (same grouping contract, launch order and
  close rule).
- **Differences**:
    - The upstream close replays the carry rows with the serial `_replay_ring`
      and folds the chunk rows one-by-one through `_accumulate` with
      `tl.where` guards; on Ascend both folds are flattened into masked
      `(8, 256)` tiles (max first, then weighted sums) — mathematically
      equivalent, vectorizer-friendly.
    - Ring/summary address arithmetic keeps the int64 width of the slot and
      ring-base tensors (vLLM slot_mapping and the ring metadata are int64
      end to end); the explicit casts are no-op guards against a future
      int32 buffer. `libdevice` comes from `tl.extra.cann`.

## Test Cases

Single-operator tests build random prefill batches (several requests,
unaligned chunk starts so regions straddle steps, `slot = -1` rows, ring
wrap-around), run `build_prefill_groups` + `update_prefill`, and compare
against a pure-torch reference that mirrors the flat close fold and the
launch order. Ring stores bit-exact; FP8 summaries with the FP8 tolerance
(>= 99% bytes bit-equal, differences within one e4m3 quantization step);
untouched summaries and ring rows must stay untouched.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_summary_update.py -k prefill
```
