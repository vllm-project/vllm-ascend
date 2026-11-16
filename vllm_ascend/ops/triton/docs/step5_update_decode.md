# Step5 update_decode (CSA region summary, decode)

## Description

- **This will be removed once triton-ascend is compatible with upstream implementations**
- **Function**: Folds one decode step's indexer rows into the Step5 CSA
  (sparse attention) state: each token's `[index_k | index_z]` row is either
  stored into its request's tail ring (open region), or — when the token's
  position is the eighth of a region (`position % 8 == 7`) — the region is
  closed and its FP8 summary is written once into the summary cache page.
  One Triton kernel per decode step, one program per token row; called from
  `AscendStep5DSAMetadata._update_summary_state`
  (`vllm_ascend/models/step5/sparse_attention.py`) on every decode forward.
- **Location**: `vllm_ascend/ops/triton/step5/summary_update.py` — kernel
  `_update_decode`, wrapper `update_decode`.
- **Formula**:
    - Ring store (open region):
      `Ring[(base + position % capacity) * 512 + {0..255}] = key`,
      `Ring[... + {256..511}] = z` (bf16).
    - Region close (online softmax over the region's 8 rows — 7 ring rows at
      `position-7 .. position-1` plus the current row, folded as one flat
      masked `(8, 256)` tile in fp32):
      `m = max(z_rows)`, `w_i = exp(z_i - m)`,
      `num = Σ w_i * key_i`, `den = Σ w_i`,
      `summary = num / max(den, 1e-20)` stored as
      `fp32 -> bf16 -> fp8 e4m3` (double rounding, byte per lane).
    - Summary page addressing (token-unit slot): byte address
      `slot // BLOCK_SIZE * MEAN_BLOCK + (slot % BLOCK_SIZE) // 8 * 256 + dim`,
      i.e. one 256-byte summary per eight-token region inside its block.
- **Algorithm flow** (processed row by row, independently):
    1. Grid `(num_decode_tokens,)`; each program loads its slot, ring base
       and position. Rows with `slot < 0` or `base < 0` (padding, dummy
       requests) do nothing.
    2. Non-closing rows store `key`/`z` into their ring row.
    3. Closing rows load the 7 earlier ring rows as a masked `(8, 256)` tile
       (`other = 0 / -inf` so masked lanes contribute zero weight), fold the
       current row in the same tile, and store the FP8 summary at the row's
       slot. The close deliberately does NOT re-store the current row into
       the ring — the region is closed and never replayed again.
- **Supported modes**: Ascend A5 only (950 series, Triton-Ascend; execution
  evidence on Ascend950DT). Called on the decode forward path including ACLGraph
  capture/replay: the grid depends only on tensor shapes and the wrapper
  branches on `slots.numel()`, both static per capture bucket.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key` | Input | This step's index K rows, `[num_tokens, 256]` (view of `[t, 1, 256]`) | bf16 | ND contiguous; row stride passed to the kernel |
| `z` | Input | This step's index Z (score) rows, same layout | bf16 | ND contiguous |
| `slots` | Input | Summary-cache slot per token (token units), `-1` skips | int64 | ND (1-D) |
| `positions` | Input | Position of each token (`seq_len - 1`) | int64 | ND (1-D) |
| `ring_base` | Input | First ring row of the token's request block (`block_table[req] * rows_per_block`), `-1` skips | int64 | ND (1-D) |
| `mean` | Output | Summary cache, `[blocks, block_size // 8, 256]` | uint8 (fp8 e4m3 payload) | ND, `stride(1) == 256`, `stride(2) == 1` |
| `ring` | Output | Tail-ring rows, flat `[total_rows * 512]` | bf16 | 1-D flat |
| `capacity` | Attribute | Ring rows per request block (`RING_CAPACITY` constexpr) | int | - |
| `MEAN_BLOCK`, `BLOCK_SIZE` | Attribute | Derived by `summary_layout(mean)`: elements between block starts, tokens per block (constexpr) | int | - |

## Constraints

- A row is a padding/dummy row iff `slot < 0` or `ring_base < 0`; such rows
  write nothing. Valid slots must be unique within a launch (decode slot
  mapping guarantees this) and inside the cache capacity.
- The close path requires the 7 earlier ring rows of the region to have been
  stored by earlier steps (or an earlier prefill chunk); the kernel does not
  zero them.
- `region_size` is fixed at 8 (Step5 `region_block_size`); the kernel
  hard-codes the `% 8` close rule and the 256-lane proxy width.
- `position % 8 == 7` with `position < 7` cannot occur (positions are
  absolute).
- Kernel launches are skipped for `slots.numel() == 0`.
- Specialization: `KEY_STRIDE`, `Z_STRIDE`, `RING_CAPACITY`, `MEAN_BLOCK`
  and `BLOCK_SIZE` are constexpr — strides are `256` for the contiguous
  `[t, 1, 256]` inputs and the cache geometry is fixed per deployment, so
  serving sees a single variant. `enable_fp_fusion=False` keeps the
  softmax fold numerically stable.
- In-place behavior: `ring` and `mean` are mutated in place (that is the
  operator's purpose); `key`/`z`/index tensors are read-only.

## Origin and Differences

- **Origin**: Adapted from the upstream Triton implementation
  `vllm.models.step5.nvidia.ops.summary_update._update_decode` /
  `update_decode` (same ring/summary layout and close semantics).
- **Differences**:
    - The upstream close path calls the serial `_replay_ring` + `_accumulate`
      helpers; on Ascend the same math is folded as one flat masked
      `(8, 256)` tile, because loops carrying 256-wide tensors with scalar
      guards inside defeat the Ascend vectorizer. The fold is a
      mathematically equivalent single-pass online softmax (max first, then
      weighted sums) instead of sequential rescaling.
    - Slot and ring-base tensors arrive as int64 (vLLM's slot_mapping and
      the ring metadata are int64 end to end), so the `slot.to(tl.int64)` /
      `base.to(tl.int64)` casts are no-op guards against a future int32
      buffer, whose late-block byte offsets could exceed 2 GiB.
    - `libdevice` comes from `tl.extra.cann` instead of `tl.extra.cuda`.

## Test Cases

Single-operator tests replay decode sequences against a pure-torch
reference: ring stores are compared bit-exact (bf16 identity); FP8 summary
bytes are decoded and checked with the FP8 summary tolerance (at least 99%
of bytes bit-equal, every differing byte within one e4m3 quantization step).
Cases cover open-row stores, closes (with pre-filled history rows), ring
wrap-around (`capacity = 16`, positions beyond capacity), skipped rows
(`slot = -1`, `base = -1`), an empty batch, and the real proxy width 256.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_summary_update.py -k decode
```
