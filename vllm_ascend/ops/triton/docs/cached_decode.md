# cached_decode

## Description

- **Function**: Refresh the graph-stable inputs of a cached pure-decode GDN layout in one NPU launch: physical state indices, accepted-token counts, speculative row masks, query boundaries and lengths, and token indices. Buffers are written in place; the wrapper returns `None`.
- **Formula**: For `i < graph_rows`: `state_output[i, :WIDTH] = state_source[i, :WIDTH]` when `i < actual_rows`, else zeros; `accepted_output[i] = accepted_source[i]` when `i < actual_rows`, else `1`; `mask_output[i] = i < actual_rows`. For `i <= graph_rows`: `query_output[i] = query_source[min(i, actual_rows)]`; `lengths_output[0] = query_output[0]` and `lengths_output[i] = query_output[i] - query_output[i-1]`. For `i < token_count`: `token_output[i] = token_source[i]`.
- **Algorithm flow** (processed row by row, independently): One program owns the complete refresh. It loops over row tiles for states/accepted/mask, then over query-boundary tiles, then over token tiles. Loads on lanes beyond `actual_rows` use the documented padding values; stores cover the full graph extent.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT products (Ascend NPU). INT32 buffers and a BOOL mask are required; the wrapper asserts an NPU device and has no CPU execution path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state_source` | Input | Current physical state indices, `[actual_rows, >= WIDTH]` | INT32 | Strided 2D |
| `query_source` | Input | Current query boundaries, `[actual_rows + 1]` | INT32 | Strided 1D |
| `accepted_source` | Input | Current accepted-token counts, `[actual_rows]` | INT32 | Strided 1D |
| `token_source` | Input | Current token indices | INT32 | Contiguous 1D |
| `state_output` | Output | Graph-owned state-index buffer | INT32 | Contiguous 2D, `[graph_rows, WIDTH]` |
| `query_output` | Output | Graph-owned query boundaries | INT32 | Contiguous 1D, `[graph_rows + 1]` |
| `accepted_output` | Output | Graph-owned accepted counts | INT32 | Contiguous 1D, `[graph_rows]` |
| `token_output` | Output | Graph-owned token indices | INT32 | Contiguous 1D, capacity `>= token_count` |
| `mask_output` | Output | Live-row mask | BOOL | Contiguous 1D, `[graph_rows]` |
| `lengths_output` | Output | Per-row query lengths | INT32 | Contiguous 1D, `[graph_rows + 1]` |
| `actual_rows` | Attribute | Number of real speculative rows | `int` | N/A |
| `graph_rows` | Attribute | Captured request extent | `int` | N/A |
| `token_count` | Attribute | Number of token indices to copy | `int` | N/A |
| `STATE_ROW_STRIDE`, `STATE_COL_STRIDE` | Attribute (constexpr) | Source state strides | `int` | N/A |
| `QUERY_STRIDE`, `ACCEPTED_STRIDE` | Attribute (constexpr) | Source scalar strides | `int` | N/A |
| `WIDTH` | Attribute (constexpr) | State columns per request | `int` | N/A |
| `BLOCK_SLOTS` | Attribute (constexpr) | Power-of-two state column tile | `int` | N/A |
| `BLOCK` | Attribute (constexpr) | Elements per tile | `int` | N/A |

## Constraints

- All inputs, outputs and the mask must reside on the same NPU; outputs are INT32 and contiguous, the mask is BOOL and contiguous.
- `0 <= actual_rows <= graph_rows <= min(state_output.shape[0], accepted_output.numel(), mask_output.numel())`.
- `actual_rows <= min(state_source.shape[0], accepted_source.numel())`; `actual_rows + 1 <= query_source.numel()`.
- `graph_rows + 1 <= min(query_output.numel(), lengths_output.numel())`, `token_source.numel() <= token_output.numel()`, and `state_source.shape[1] >= WIDTH`.
- Padding conventions are part of the contract: zero state columns and accepted count `1` for graph rows without a live request; the live mask is explicit.
- `actual_rows`, `graph_rows` and `token_count` are `do_not_specialize`, so batch changes inside one graph do not recompile.

## Origin and Differences

- **Origin**: Developed from scratch for Ascend. It replaces the per-group host slicing/copy sequence used when a pure-decode layout cache is hit.
- **Differences**: One launch refreshes every captured input with fixed padding semantics, avoiding per-buffer H2D copies and host synchronization. The layout, column width and strides stay captured; only the runtime row counts change.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_cached_decode.py
```

The test compares every output with a torch golden using exact equality; it covers equal and smaller live batches, empty token sets, nonzero query offsets, width `1`, and rows beyond the 32-element tile boundary.

## Example

This is a pure data-movement refresh of graph-owned inputs, so all comparisons are bit-exact:

```text
query_output = query_source[:actual_rows] padded by repeating its last entry
lengths_output = diff(query_output, prepend=0)
state/accepted rows beyond actual_rows are zero / one
```
