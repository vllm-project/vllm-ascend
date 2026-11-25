# aligned_indices

## Description

`ctx.compute_aligned_state_indices(seq_lens, num_reqs)` gathers physical Mamba state IDs for all cache
groups in one Ascend Triton launch. It writes into the output buffer owned by
the vLLM context and returns the view `output[:, :num_reqs]`. An empty batch
returns the empty view without launching a kernel.

For each group, live row and state slot, the result is

```python
first_slot = max((seq_lens[row] - 1) // cache_block_size, 0)
output[group, row, slot] = table[group][row, first_slot + slot]
```

## Parameters

| Parameter | Type and layout | Use |
| --- | --- | --- |
| `ctx.aligned_index_block_table_ptrs` (or `ctx.block_table_ptrs` on older contexts) | Contiguous INT64 vector, one pointer per group | Addresses of the groups' INT32 block tables |
| `seq_lens` | INT32 or INT64 vector, possibly strided | Current sequence lengths for the live rows |
| `ctx.aligned_state_indices` | Contiguous INT32 tensor, 32-byte aligned, shaped `[num_groups, max_reqs, state_slots]` | Caller-owned destination. Rows beyond `num_reqs` retain their contents. |
| `num_reqs` | Integer from zero to `min(max_reqs, len(seq_lens))` | Number of live rows |
| `ctx.block_table_stride_req` | Integer, in elements | Shared physical row stride. Columns are contiguous. Each selected slot must be within the logical table width. |
| `ctx.block_size` | Positive integer | Tokens per physical cache block |

All tensors reside on the same NPU. For graph replay the caller keeps the table
and buffer addresses stable and updates their contents in place.

## Ascend implementation

The kernel partitions the contiguous output into tiles of 32 INT32 values.
Each tile owns complete 32-byte spans, including spans shared by adjacent cache
groups. It reads one scalar block-table pointer for each group intersecting the
tile, gathers the selected INT32 values, and writes only live rows. The grid is capped at 65535
programs. Larger outputs assign multiple tiles to each program. The live batch
count is a runtime argument marked `do_not_specialize`.

Upstream's `get_aligned_state_indices_multi_group_kernel` loads a vector of
table pointers. On A3 with Triton 3.6.0, compiling that kernel for 24 groups,
8 requests and 4 slots fails in `TritonToUnstructure` during pointer-offset
analysis. The Ascend kernel keeps the same index formula using scalar table
pointers.

## Tests

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_aligned_indices.py
pytest -q tests/e2e/pull_request/one_card/test_gdn_metadata_reuse_v2.py -k aligned_state_indices
```

The tests compare INT32 IDs bit for bit, check inactive rows, and replay
captured work with updated lengths and table contents. Single-card validation
covered Ascend910_9382 (A3) and Ascend950PR_957c (A5) with Triton 3.6.0.
