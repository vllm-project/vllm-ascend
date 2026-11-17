# `_compute_slot_mappings_kernel`

## Overview

`_compute_slot_mappings_kernel` converts every scheduled token's logical
position into the physical KV-cache slot consumed by the attention and
cache-write paths. It is launched by
`AscendBlockTables.compute_slot_mappings` from
`vllm_ascend/worker/v2/block_table.py`.

The grid is `(num_kv_cache_groups, num_reqs + 1)`. The final program on the
request axis fills the unused output tail with `PAD_ID` for graph-capture
compatibility.

## Block-table layout

`BlockTables` distinguishes two block sizes:

- `block_sizes`: the KV-cache manager's allocation block size.
- `kernel_block_sizes`: the size used to index slot mappings and the attention
  kernel's expanded block table.

The upstream `BlockTables` implementation passes both sizes to this kernel:
`block_sizes_tensor` contains allocation block sizes and
`kernel_block_sizes_tensor` contains kernel block sizes. If a KV block is split
into several kernel blocks, the block-table row has a corresponding entry for
each kernel block.

For a non-CP launch, with `kernel_block_size = B` and position `p`:

```text
block_index = p // B
block_offset = p - block_index * B
slot_id = block_table[request, block_index] * B + block_offset
```

The multiply/subtract form is equivalent to `p % B`; it avoids the scalar
remainder lowering that is slow on Ascend.

## Context parallelism

With `CP_SIZE > 1`, a virtual block has size
`kv_block_size * CP_SIZE`. The kernel first finds the virtual block and its
in-block offset. Interleaved chunks are assigned to ranks using
`CP_INTERLEAVE`; a position not owned by `cp_rank` receives `PAD_ID`. The
rank-local position is then converted to a `kernel_block_size` block-table
index and offset. This distinction is required when one allocation block is
split into multiple kernel blocks.

For a local position, the interleaved offset is compacted into a rank-local
offset before forming the final slot ID. This follows the V0.28.0 upstream
`_compute_slot_mappings_kernel` mapping rule.

## Bounded window gather

For each `TRITON_BLOCK_SIZE` token tile, block indices are close together. The
kernel therefore:

1. Finds the tile's smallest valid block index.
2. Loads a contiguous window from that request's block-table row.
3. Uses `tl.gather` with relative block indices to obtain physical block IDs.

`BLOCK_TABLE_WINDOW_SIZE` is a power-of-two constexpr selected by the worker:

```text
next_power_of_2(ceil(TRITON_BLOCK_SIZE / min(kernel_block_sizes)) + 1)
```

It is derived from kernel block sizes, not allocation block sizes. The extra
entry safely handles a tile beginning near a block boundary. The runtime load
mask uses the row stride, so the load never crosses the allocated block-table
row. This bounds UB use independently of request length; unlike full-row
staging, a long block-table row does not require a long UB allocation.

An empty request has `start_idx == end_idx`. Its token loop has zero
iterations, so its `idx_mapping` sentinel is never used to form a block-table
address.

## Main inputs and attributes

| Name | Meaning |
| --- | --- |
| `idx_mapping` | Maps each scheduled request to its persistent block-table row. |
| `query_start_loc` | Prefix offsets that delimit each request's token range. |
| `pos` | INT64 logical token positions; converted to INT32 for kernel arithmetic. |
| `block_table_ptrs` | One INT32 block-table base pointer per KV-cache group. |
| `block_table_strides` | Row stride of each group block table, in entries. |
| `block_sizes` | KV-cache allocation block sizes from upstream `block_sizes_tensor`. |
| `kernel_block_sizes` | Kernel block sizes from upstream `kernel_block_sizes_tensor`. |
| `CP_SIZE`, `CP_INTERLEAVE`, `cp_rank` | Context-parallel ownership configuration. |
| `TRITON_BLOCK_SIZE` | Token tile size; production launch uses 1024. |
| `BLOCK_TABLE_WINDOW_SIZE` | Power-of-two upper bound for the staged tile window. |

## Constraints and tests

- Positions must fit in INT32 after conversion.
- `query_start_loc` must be non-decreasing and its final value must not exceed
  the slot-mapping output capacity.
- Valid positions must map to block indices within the associated request's
  block-table row.
- `BLOCK_TABLE_WINDOW_SIZE` must cover the maximum span of block indices in a
  token tile for every group.

The NPU test compares this kernel against V0.28.0 upstream for CP=1, CP=2, and
CP=4. It covers empty requests, cross-block and cross-tile ranges, multiple
kernel block sizes, padding, and the `out` interface:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py
```
