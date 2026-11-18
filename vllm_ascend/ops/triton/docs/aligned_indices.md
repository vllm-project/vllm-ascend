# aligned_indices

## Description

- **Function**: Gather each request's physical Mamba state slots from per-group block tables into caller-provided storage (`GDN` metadata reuse). Pure integer data movement: no arithmetic on values, no host synchronization.
- **Formula**: For group `g`, request `r < num_reqs` and slot `s < STATE_SLOTS`:
  `first_slot_r = max((seq_lens[r] - 1) // CACHE_BLOCK_SIZE, 0)`,
  `window_start_r = first_slot_r // 8 * 8`,
  `output[g, r, s] = table_g[r, window_start_r + min(s, STATE_SLOTS - 1)]`.
  Requests where the source column is outside `TABLE_COLS` are masked and never read. Output rows `>= num_reqs` are not written. The flat fallback uses `first_slot_r + s` directly.
- **Algorithm flow** (processed row by row, independently): The wrapper validates the layout and selects one of two kernels. The window kernel assigns row tiles to programs, loads one bounded `WINDOW`-wide slice per row (only the state-slot window enters UB), gathers locally with an fp32 bitcast gather, and stores masked output slots. The flat kernel walking output offsets reconstructs `(group, row, slot)` arithmetically and visits only the groups intersecting each 32-byte output span.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT products (Ascend NPU). Output is a 3-D INT32 tensor; CPU is not a supported execution device.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `table_ptrs` | Input | Device pointer to each group's block table | INT64 | Contiguous 1D, `[num_groups]` |
| `seq_lens` | Input | Sequence length per request, at least `num_reqs` entries | INT32 / INT64 | Strided 1D |
| `output` | Input/Output | Gathered state slots; rows below `num_reqs` overwritten, other rows preserved | INT32 | Contiguous 3D, `[num_groups, max_reqs, state_slots]`, 32-byte aligned |
| `num_reqs` | Attribute | Real request count | `int` | N/A |
| `table_stride` | Attribute | Row stride of each physical table | `int` | N/A |
| `table_cols` | Attribute | Logical table columns used by the column mask | `int` | N/A |
| `cache_block_size` | Attribute | Tokens per physical cache block | `int` | N/A |
| `TABLE_COLS` | Attribute (constexpr) | Column bound for masked source loads | `int` | N/A |
| `CACHE_BLOCK_SIZE` | Attribute (constexpr) | Tokens per block in the kernel | `int` | N/A |
| `STATE_SLOTS` | Attribute (constexpr) | Slots gathered per request | `int` | N/A |
| `BLOCK_SLOTS` | Attribute (constexpr) | Power-of-two slot tile | `int` | N/A |
| `BLOCK_ROWS` | Attribute (constexpr) | Rows per tile in the window kernel | `int` | N/A |
| `WINDOW` | Attribute (constexpr) | Power-of-two source-column window | `int` | N/A |
| `TILES_PER_PROGRAM` | Attribute (constexpr) | Row tiles handled by one program | `int` | N/A |
| `MAX_REQS`, `NUM_GROUPS`, `BLOCK` | Attribute (constexpr) | Flat-kernel geometry | `int` | N/A |

## Constraints

- `output` is INT32, contiguous, 32-byte aligned, and shares its device with `table_ptrs` and `seq_lens`; inactive output rows stay unchanged.
- `state_slots <= table_cols <= table_stride`; table columns are contiguous and rows may be physically padded.
- `0 <= num_reqs <= min(max_reqs, seq_lens.shape[0])`; `seq_lens` is INT32 or INT64.
- `table_ptrs` is an INT64 contiguous vector with one device pointer per group.
- The window path requires `max_reqs * state_slots % 8 == 0` and a tile that fits the UB budget; otherwise the flat path runs. Both keep the total grid at or below 65535 programs.
- Graph replay requires stable table addresses; the caller pins them before capture. `num_reqs` is `do_not_specialize`, so a changing batch does not recompile.

## Origin and Differences

- **Origin**: Developed from scratch for Ascend, following the Triton-Ascend gather tutorial. The state-selection semantics follow vLLM's GDN metadata contract (vLLM #58737 requires block-table rows to match `seq_lens` rows).
- **Differences**: Ascend gather supports FP32 but not INT32, so physical IDs are bitcast through FP32 and back. Only a bounded aligned column window enters UB, and launch geometry (window vs flat) is chosen on the host. One launch serves every cache group.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_aligned_indices.py
```

The test compares against a torch golden with `torch.equal` (bit-exact) and covers window/flat dispatch, tail rows, INT32/INT64 lengths, non-contiguous table rows and the empty batch.

## Example

Input tables are pure index data; the output is a gather of the block IDs to be consumed by graph-stable state updates, so exact comparison applies:

```text
first_slot = max((seq_len - 1) // cache_block_size, 0)
output[group, row, slot] = table[group][row, first_slot + slot]
```
