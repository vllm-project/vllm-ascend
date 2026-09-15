# chunk_local_cumsum

## Description

- **Function**: Partitions the sequence dimension into chunks and independently computes a cumulative sum inside each chunk. Reverse accumulation, output scaling, head-first layout, and variable-length sequences are supported.
- **Formula**: Computes an independent prefix sum within each sequence chunk:
- Input `g`: `[B, T, H]` when `head_first=False`, or `[B, H, T]` when `head_first=True`
- Forward accumulation: `output[b, t, h] = sum(g[b, chunk_start:t + 1, h])`
- Reverse accumulation: `output[b, t, h] = sum(g[b, t:chunk_end, h])` when `reverse=True`
- Optional scaling: `output = scale * output` when `scale` is provided
- Output `output`: same shape as `g`, with each chunk accumulated independently
- **Algorithm flow**:
  1. Map each Triton program to a sequence chunk and head tile.
  2. Load the valid input elements, respecting `cu_seqlens` in variable-length mode.
  3. Perform a forward or reverse prefix sum within the chunk.
  4. Apply the optional scale, cast to `output_dtype`, and store the result in the input layout.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel); fixed-length and variable-length sequences; sequence-first and head-first layouts; eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `g` | Input | Input `[B, T, H]` if `head_first=False`, otherwise `[B, H, T]` | fp32 | ND |
| `chunk_size` | Input (attribute) | Number of tokens per chunk; must be a power of two | int32 | scalar |
| `reverse` | Input (attribute) | Compute a reverse cumulative sum when `True`; default `False` | bool | scalar |
| `scale` | Input (attribute) | Optional output scale; default `None` | fp32 | scalar |
| `cu_seqlens` | Input | Cumulative sequence lengths `[N + 1]` | int32 | ND |
| `head_first` | Input (attribute) | Select `[B, H, T]` layout when `True`; default `False` | bool | scalar |
| `output_dtype` | Input (attribute) | Requested output dtype; default `torch.float` | torch dtype | scalar |
| `output` | Output | Chunk-local cumulative sum with the same shape as `g` | specified by `output_dtype` | ND |

## Constraints

- `chunk_size` must be a power of two; otherwise, an `AssertionError` is raised.
- Only three-dimensional inputs are supported; four-dimensional inputs are not supported.
- Batch size must be 1 when `cu_seqlens` is provided.
- Empty or non-contiguous inputs are not supported; the last dimension must be contiguous.
- The default fp32 output helps prevent intermediate overflow during backward propagation and context-parallel execution.

## Origin and Differences

- **Origin**: Based on the chunk-local cumulative-sum implementation from the flash-linear-attention project (MIT license; see the source-file header).
- **Differences**:
    - Adapted to Ascend NPU Triton execution and its supported tensor layouts.
    - Adds vLLM Ascend validation and variable-length metadata handling for inference workloads.

## Test Cases

The test compares forward and reverse accumulation with a PyTorch reference across chunk sizes, scaling options, layouts, variable-length inputs, and tail chunks.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_local_cumsum.py
```
