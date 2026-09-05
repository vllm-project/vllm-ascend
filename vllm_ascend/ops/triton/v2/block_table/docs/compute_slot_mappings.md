# _compute_slot_mappings_kernel

## Description

- **Location**: `vllm_ascend/ops/triton/v2/block_table/compute_slot_mappings.py`
  — `_compute_slot_mappings_kernel`, launched by
  `AscendBlockTables.compute_slot_mappings`.
- **Function**: Computes the physical KV-cache slot ID for every scheduled
  token in MRV2. KV cache uses block-based physical storage, while each token
  is identified by its logical position in a request. The request's
  `block_table` maps the logical block index to a physical cache block; this
  kernel combines that physical block number with the token's in-block offset
  to produce the slot ID consumed by attention and cache-write operators. It
  supports multiple KV-cache groups and context parallelism (CP), and fills
  the unused tail of each output row with `PAD_ID`.
- **Formula**: For a token at logical position `p`, let
  `slice_size = block_size * CP_SIZE`,
  `block_index = p // slice_size`, and
  `block_offset = p % slice_size`. The physical block is read from the
  request's block table. When `CP_SIZE == 1`, the output is
  `physical_block * block_size + block_offset`. With CP enabled, only offsets
  assigned to `cp_rank` are retained; their rank-local offset is
  `(block_offset // (CP_INTERLEAVE * CP_SIZE)) * CP_INTERLEAVE +
  block_offset % CP_INTERLEAVE`. Other ranks' offsets produce `PAD_ID`.
- **Context-parallel mapping**: With `CP_SIZE > 1`, one virtual slice contains
  `CP_SIZE` physical blocks and has size `block_size * CP_SIZE`. For a logical
  position `p`:
  1. `block_index = p // (block_size * CP_SIZE)` selects the virtual slice and
     therefore the block-table entry.
  2. `block_offset = p % (block_size * CP_SIZE)` selects a position inside the
     virtual slice.
  3. `(block_offset // CP_INTERLEAVE) % CP_SIZE` identifies the rank that owns
     the position. Positions owned by another rank are represented by
     `PAD_ID` on the current rank.
  4. Complete interleave rounds and the remainder within one interleave segment
     are combined into a contiguous rank-local offset. The final slot ID is
     `physical_block * block_size + local_offset`.
- **Algorithm flow**:
  1. Launch a two-dimensional grid over KV-cache groups and requests, plus one
     padding program per group.
  2. Resolve the selected request through `idx_mapping`, then read its token
     interval from `query_start_loc`.
  3. For block-table rows up to the validated staging limit, load the selected
     request's row once with a contiguous GM load. Larger rows use per-token
     direct GM loads so the compiler does not allocate an unbounded staged row
     in UB. Process the interval in `TRITON_BLOCK_SIZE` tiles, convert positions
     to INT32, and calculate block indices and offsets.
  4. Calculate slot IDs directly for non-CP execution. For CP execution,
     convert virtual offsets to rank-local offsets and replace non-local slots
     with `PAD_ID`.
  5. The extra request program fills `[actual_num_tokens, max_num_tokens)` with
     `PAD_ID` for each KV-cache group.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. The kernel is used
  by `AscendBlockTables.compute_slot_mappings` in the MRV2 model runner and
  supports eager and graph-capture execution with a fixed grid and compile-time
  CP attributes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `max_num_tokens` | Input | Number of allocated entries in each output row; entries after the actual token count are padded. | INT | scalar |
| `idx_mapping` | Input | Maps each scheduled request index to a row in every group's block table; shape `[num_reqs]`. | INT32 | ND |
| `query_start_loc` | Input | Prefix offsets delimiting each request's token interval; shape `[num_reqs + 1]`. The last value is the actual token count. | INT32 | ND |
| `pos` | Input | Logical sequence position for every scheduled token; shape `[actual_num_tokens]`. Values are converted to INT32 inside the kernel. | INT64 | ND |
| `block_table_ptrs` | Input | Device pointer table containing one INT32 block-table base address per KV-cache group; shape `[num_groups]`. | UINT64 | ND |
| `block_table_strides` | Input | Row stride, in elements, of each group's block table; shape `[num_groups]`. | INT64 | ND |
| `block_sizes` | Input | Physical KV-cache block size for each group; shape `[num_groups]`. | INT32 | ND |
| `slot_mappings_ptr` | Output | Physical slot IDs, with one row per KV-cache group; shape `[num_groups, max_num_tokens]`. | INT32 | ND |
| `slot_mappings_stride` | Input (attribute) | Row stride, in elements, of `slot_mappings_ptr`. | INT | scalar |
| `cp_rank` | Input (attribute) | Rank of the current device in the context-parallel group. | INT | scalar |
| `CP_SIZE` | Attribute | Number of ranks in the context-parallel group. | constexpr INT | scalar |
| `CP_INTERLEAVE` | Attribute | Number of consecutive token positions assigned to one CP rank before interleaving to the next rank. | constexpr INT | scalar |
| `PAD_ID` | Attribute | Value written for padding and token positions owned by another CP rank; production uses `PAD_SLOT_ID`. | constexpr INT | scalar |
| `TRITON_BLOCK_SIZE` | Attribute | Number of token positions processed per loop tile; production uses 1024. | constexpr INT | scalar |
| `BLOCK_TABLE_PAD_SIZE` | Attribute | Power-of-two upper bound for the allocated row stride. The load is masked by the runtime row stride. | constexpr INT | scalar |
| `USE_BLOCK_TABLE_STAGING` | Attribute | Selects the contiguous staged-row path for bounded rows; larger rows use direct block-number loads. | constexpr BOOL | scalar |

## Constraints

- The launch grid must be `(num_groups, num_reqs + 1)`. The final program on
  the request axis is reserved for output padding.
- `idx_mapping` and `query_start_loc` must be INT32. `pos` is supplied as INT64
  by MRV2, and every position must fit in INT32 because the optimized kernel
  explicitly narrows it before index arithmetic.
- Each `block_table_ptrs` entry must be a valid device address for an INT32
  two-dimensional block table and must remain alive for the duration of the
  launch. `block_table_strides` and `block_sizes` must describe the matching
  group in the same order.
- `query_start_loc` must be non-decreasing, start at zero, and end at a value no
  greater than `max_num_tokens`. Every value in `idx_mapping` and every derived
  block index must be within the corresponding block-table bounds.
- `CP_SIZE >= 1`, `0 <= cp_rank < CP_SIZE`, and `CP_INTERLEAVE >= 1`.
- `slot_mappings_ptr` must be INT32 with at least `max_num_tokens` entries per
  group. `TRITON_BLOCK_SIZE` must be a positive compile-time constant.
- Every derived block index must be smaller than its group's runtime row stride,
  which in turn must not exceed `BLOCK_TABLE_PAD_SIZE`.
- `USE_BLOCK_TABLE_STAGING` may only be enabled when the staged row and the
  per-token temporaries fit the target UB. The host currently enables it for
  power-of-two row bounds up to 16K entries, the largest size validated by the
  original optimization on Ascend A3.
- Graph capture requires the number of groups, request capacity, output
  capacity, and compile-time CP attributes to remain compatible with the
  captured launch.

## Origin and Differences

- **Origin**: Adapted from
  `vllm.v1.worker.gpu.block_table._compute_slot_mappings_kernel`, the upstream
  vLLM kernel used by the MRV2 block-table path.
- **Differences**:
    - Converts logical positions from INT64 to INT32 before block-index,
    block-offset, and address calculations, reducing INT64 scalar arithmetic
    on Ascend NPU.
    - Stages one complete bounded request row with a contiguous masked GM load
    and uses `tl.gather` for token-to-block lookup. Rows above the validated
    staging limit preserve the upstream direct-load algorithm to bound UB use.
    - Replaces the INT32 remainder used for the block offset with
    multiply/subtract to avoid scalar fallback on Ascend.
    - Preserves the upstream launch grid, CP mapping, padding behavior, and
    optional `out` semantics.
    - The kernel is launched by the Ascend-specific
    `AscendBlockTables.compute_slot_mappings` override and writes to the
    existing INT32 slot-mapping buffer required by the Ascend KV-cache path.

## Test Cases

The single-operator test compares the Ascend kernel bit-for-bit with the
upstream MRV2 kernel for two KV-cache groups and CP configurations
`(size, rank, interleave) = (1, 0, 1), (2, 1, 2), (4, 2, 1)`. It also exercises
both the staged-row boundary and a 131072-entry direct-load row, then exercises
the `AscendBlockTables.compute_slot_mappings(..., out=...)` override and checks
the returned view, physical slot IDs, and padded tail.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py
```
