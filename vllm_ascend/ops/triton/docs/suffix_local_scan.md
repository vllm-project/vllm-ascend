# Suffix local scan

## Description

- **Function**: Find the longest, rightmost earlier match in each request's history.
- **Formula**: `score = match_length * Lp1 + end_position`; reduce the maximum per scan block.
- **Algorithm flow**: `launch_suffix_local_scan` scans active request rows and writes packed scores for finalize.
- **Supported modes**: Ascend A3 and A5; eager and graph execution.

## Parameters

`B` is the active batch size, `R` the request capacity, `L` the history capacity and `N` the scan block count.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `token_ids_ptr`, `token_ids_stride` | Input / Attribute | Dense history and row stride | int32, integer | `[R, L]`, scalar |
| `idx_mapping_ptr`, `total_len_ptr`, `num_sampled_ptr` | Input | Request slots, history lengths and sampled counts | int32 | `[B]`, `[R]`, `[B]` |
| `scratch_ptr`, `scratch_stride` | Output / Attribute | Packed best match per block and row stride | int64, integer | `[B, N]`, scalar |
| `Lp1`, `MAX_DEPTH`, `BLOCK`, `launch_options` | Attribute | Position radix, match-depth limit, scan tile and compiler options | Integers / mapping | Host values |

## Constraints

Histories and scratch have contiguous columns; the wrapper reads their actual row strides.
`Lp1` exceeds every history position. Depth and block width are positive; token IDs are nonnegative.
The launch grid uses `idx_mapping_ptr.shape[0]`, rather than buffer capacity.

## Origin and Differences

- **Origin**: [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
- **Differences**: Keep indirect-load masks separate from loop-carried match state for Triton-Ascend lowering.

## Test Cases

Packed integer scores must exactly match the CPU reference, including empty histories, block boundaries and rightmost ties.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_suffix_local_scan.py
```

## Example

N/A for standalone use; the speculator calls this wrapper through the Triton dispatcher.
