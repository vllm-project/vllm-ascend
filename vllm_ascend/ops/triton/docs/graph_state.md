# graph_state

## Description

- **Function**: Batch group-local GDN graph-input writes into captured destination buffers in one launch, with padding and an optional clear-table fill. `GDNGraphStateUpdater.apply(updates)` writes in place and returns `None`; it caches the packed pointer table across steps.
- **Formula**: For each group `g`: `destination_g[i] = source_g[i]` for `i < actual_rows`; `destination_g[i] = PAD_VALUE` for `actual_rows <= i < graph_rows`. When a clear table is present, `clear_g[i] = PAD_SLOT_ID` for `i < graph_rows`. Supports 1-D destinations and 2-D destinations with `WIDTH` columns.
- **Algorithm flow** (processed row by row, independently): `apply` validates every group before submitting work, builds a pinned `(source, destination, clear)` pointer vector for the current signatures, and launches one program per group. Each program tiles `graph_rows * WIDTH`, loads source rows below `actual_rows` (other lanes use `PAD_VALUE`), stores every graph row and optionally fills the clear table. The CPU fallback performs the same copies and fills with torch. Zero-argument callables in `updates` run after every write.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT products (Ascend NPU) through the Triton kernel; a CPU fallback exists for CPU tensors. Sources and destinations are INT32.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `updates` | Input/Output | One tuple per group: `(source, destination, actual_rows, graph_rows, padding, clear)`; optional zero-argument finalizers may follow | Python objects | List |
| `source` | Input | Group-local cache indices | INT32 | Contiguous 1D or 2D, `[>= actual_rows, >= WIDTH]` |
| `destination` | Output | Captured graph input buffer | INT32 | Contiguous 1D or 2D, `[>= graph_rows, WIDTH]` |
| `clear` | Output | Optional buffer cleared to `PAD_SLOT_ID` | INT32 / `None` | Contiguous 2D, `[>= graph_rows, CLEAR_WIDTH]` |
| `actual_rows` | Attribute | Live rows copied from the source | `int` | N/A |
| `graph_rows` | Attribute | Captured row extent | `int` | N/A |
| `padding` | Attribute | Value written to padded source rows | `int` | N/A |
| `pointers` | Attribute | Packed INT64 pointer table built by the wrapper | INT64 | Contiguous 1D, `[3 * num_groups]` |
| `WIDTH` | Attribute (constexpr) | Destination columns (1 for 1-D) | `int` | N/A |
| `SOURCE_ROW_STRIDE`, `SOURCE_COL_STRIDE` | Attribute (constexpr) | Source strides | `int` | N/A |
| `PAD_VALUE` | Attribute (constexpr) | Padding value for rows without a request | `int` | N/A |
| `CLEAR_WIDTH`, `CLEAR_VALUE` | Attribute (constexpr) | Clear-table width and value (`PAD_SLOT_ID`) | `int` | N/A |
| `BLOCK` | Attribute (constexpr) | Elements per write tile | `int` | N/A |

## Constraints

- Destinations are INT32, contiguous and distinct; all tensors in one `apply` share device, dtype, dimensionality and source strides.
- `0 <= actual_rows <= graph_rows <= destination rows`; the source has at least `actual_rows` rows and at least `WIDTH` columns.
- `clear` is present for every group or for none; its row capacity covers `graph_rows` and its width is constant.
- The packed pointer tensor is rebuilt only when pointer signatures change and retains the referenced allocations; the launch grid is the group count.
- `actual_rows` and `graph_rows` are `do_not_specialize`; `graph_rows == 0` skips the kernel but still runs queued finalizers.
- The wrapper performs no device-to-host synchronization.

## Origin and Differences

- **Origin**: Developed from scratch for Ascend to replace per-group eager torch copies in `GDNAttentionMetadataBuilder.update_block_table()`.
- **Differences**: One batched launch handles every cache group, pins the address table once per signature, writes padding explicitly and can clear the retired speculative state table. GPU upstream uses per-group tensor operations.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_graph_state.py
```

The test compares destinations and clear tables with torch goldens using exact equality; it covers 1-D/2-D layouts, partial batches, custom padding, clear tables, multiple groups and a zero-row update.

## Example

Graph-state writes only move index data, so exact comparison applies:

```text
destination[:actual_rows] = source[:actual_rows]
destination[actual_rows:graph_rows] = padding
clear[:graph_rows] = PAD_SLOT_ID
```
