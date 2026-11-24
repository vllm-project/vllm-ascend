# `_compute_slot_mappings_kernel`

## Description

- **Function**: Maps scheduled token positions to flat KV-cache slots for the
  V2 Model Runner. The shared `AscendBlockTables.compute_slot_mappings`
  wrapper launches the kernel. Public block tables contain manager block IDs.
  Each `AttentionGroup` constructs its kernel-granularity table when building
  attention metadata.
- **Formula**: For manager block size `B` and rank-local position `p`,
  `block_index = p // B`, `offset = p - block_index * B`, and
  `slot = block_table[request, block_index] * B + offset`.
- **Algorithm flow**:
  1. For a context-parallel sharded group, find the virtual block of size
     `B * CP_SIZE`, assign interleaved positions to ranks, and compact each
     rank's positions before looking up the manager block. Replicated groups
     use the original token position.
  2. For each token tile, stage a bounded window of the request's block-table
     row and gather its manager block IDs. The wrapper selects the window as
     `next_power_of_2(ceil(TRITON_BLOCK_SIZE / min(block_sizes)) + 1)`.
     The extra entry covers tiles starting near a block boundary, and the
     load mask bounds the window to the row stride.
  3. Emit `PAD_ID` for tokens owned by another rank, dummy requests, disabled
     groups, and unused output capacity. The last program on the request axis
     pads the output tail for graph capture.
- **Supported modes**: The single-card test targets Atlas A2, Atlas A3, and
  Ascend A5. The recorded hardware validation for this change is on Atlas A3.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `idx_mapping` | Input | Persistent block-table row per scheduled request. `-1` marks a dummy request. | `int32` or `int64` | 1-D tensor |
| `query_start_loc` | Input | Cumulative token offsets, with one extra entry for the total token count. | int32 | 1-D tensor |
| `pos` | Input | Logical positions of scheduled tokens, converted to int32 in the kernel. | int64 | 1-D tensor |
| `block_table_ptrs`, `block_table_strides` | Input | Base addresses of the int32 manager block tables and their row strides in entries, one per KV-cache group. | uint64 addresses, int64 strides | 1-D tensors |
| `block_sizes` | Input | Manager block size for each KV-cache group. | int32 | 1-D tensor |
| `dcp_sharded` | Input | Per-group flag selecting rank-local positions for sharded KV. Replicated groups keep global positions. | bool | 1-D tensor |
| `slot_mapping_enabled` | Input | Per-group flag enabling token-position slot mapping, used when `HAS_SLOT_MAPPING_ENABLED` is true. | bool | 1-D tensor or None |
| `slot_mappings_ptr` | Output | Preallocated flat slot IDs. The wrapper writes its persistent buffer or the caller's `out` in place, and returns the requested output prefix. | int32 | 2-D tensor with contiguous columns |
| `max_num_tokens`, `slot_mappings_stride` | Attribute | Output capacity per group and output row stride in entries. | int | Scalar |
| `cp_rank`, `CP_SIZE`, `CP_INTERLEAVE` | Attribute | Local context-parallel rank, number of ranks, and positions assigned per rank in each interleave round. | int | Scalar |
| `PAD_ID` | Attribute | Sentinel for slots that do not correspond to local live tokens. | int | Scalar |
| `TRITON_BLOCK_SIZE`, `BLOCK_TABLE_WINDOW_SIZE` | Attribute | Token tile size (1024 in production) and power-of-two staging window. | int | Scalar |
| `HAS_SLOT_MAPPING_ENABLED` | Attribute | Enables the per-group slot-mapping flag. | bool | Scalar |

## Constraints

- Positions and resulting slot IDs must fit int32. The staged gather converts
  manager block IDs through float32, so block IDs must be exactly
  representable in float32.
- `query_start_loc` must be non-decreasing, and its final value must be within
  the output capacity. The wrapper's `num_tokens_padded` selects the return
  view without changing the allocated capacity. An empty request has no
  token iterations.
- Each active token must resolve to a valid entry in its manager block-table
  row. A dummy request emits padding with its block-table loads masked.
- The staged window must cover the maximum span of block indices in a tile.
  Positive manager block sizes, rank counts, and interleave sizes are required.
- Graph capture and replay consume the preallocated slot buffer. Only the
  output buffer is written. For state-only groups, disabling token slot
  mapping fills the corresponding output with `PAD_ID`.

## Origin and Differences

- **Origin**: vLLM V2's GPU block-table slot-mapping implementation. Ascend
  uses an int32 output and stages a bounded window for NPU gather.
- **Differences in this refactor**: The slot kernel reads the manager block
  size for both table lookup and slot arithmetic. Kernel-sized block table
  expansion happens in `AttentionGroup` instead of the shared block table.

## Test Cases

On Atlas A3, the DeepSeek-V4-Flash-0731 W8A8 V2 runner (DP2 × PCP4,
TP1, 7-step DSpark) called this operator for a 79-token prefill with
`idx_mapping.shape=(1,)`, `query_start_loc.shape=(2,)`, and
`pos.shape=(79,)`. The captured slot-mapping launch used `CP_SIZE=1`,
a token tile of 1024, a gather window of 1024, and per-group manager
block sizes `[128, 4096, 32, 32, 2, 8]`. The baseline passed the same
values as its per-group kernel block sizes for this configuration
(vLLM-Ascend main `de6351f5`, PR `98d8a92c`, vLLM `c410543d9`).

The single-card NPU suite replays this input shape and the six manager
block sizes with synthetic block IDs. It compares every slot ID against
an independent host reference. The other cases cover block sizes 128 and 256,
CP sizes 1, 2, and 4, sharded and replicated groups, requests crossing block
and tile boundaries, empty requests, and padding. It also checks the
caller's preallocated `out` buffer and its padded tail.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_slot_mapping.py
```
