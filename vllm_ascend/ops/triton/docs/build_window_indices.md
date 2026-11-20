# build_window_indices_triton

## Description

- Function. Build a fixed-width row of logical KV positions and a valid count
  for each query in causal sliding-window attention (SWA). The A5 mixed-quant
  sparse attention operator consumes the first `length` entries of each row.
- Formula. For query position `p` and window size `W`, the ordinary count is
  `base = min(p + 1, W)`. When a visible count `v` is supplied, the count is
  `length = min(base, max(v, 0))`. Otherwise, `length = base`.
  The first valid position is `start = p + 1 - length`.
- Algorithm flow. One Triton program handles one query row. It computes the
  count, writes `start + column` in the first `length` columns, fills the
  remaining columns with `-1`, and writes the same count to the length tensor.
- Supported modes. Ascend A5, on the DeepSeek-V4.1 packed-cache attention path.

During bounded replay, the metadata builder computes a visible count per real
query using the request's replay boundary `r`. The first readable position is
`max(0, p - W + 1, r)`, and the count is `p - first_readable_position + 1`
for a query in the replayed range. Applying that count before generating the
row makes the first entry agree with the replay boundary.

## Parameters

`T` is the number of input rows, including any padding, and `W` is the
window width. The function returns `(indices, lengths)`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `positions` | Input | Logical query positions `[T]`, flattened in batch order. | INT64 | ND |
| `window_size` | Attribute | Positive window width `W`, including the current token. | Python int | Scalar |
| `visible_lengths` | Optional input | One visible KV count per real query row, in the same order as `positions`. Defaults to `None`. | INT32 | ND, `[T_real]` or `[T_real, 1]` |
| `indices_output` | Optional output | Caller-provided buffer for the returned indices. A buffer is allocated when omitted. | INT32 | ND, `[T, 1, W]` |
| `lengths_output` | Optional output | Caller-provided buffer for the returned valid counts. A buffer is allocated when omitted. | INT32 | ND, `[T, 1]` |

## Constraints

- Supply non-negative logical token positions representable as INT32 and a
  positive `window_size`. Empty input returns empty outputs without a kernel
  launch.
- When `visible_lengths` is supplied, the counts describe the leading real
  query rows. Rows beyond `visible_lengths.numel()` have length zero and all
  indices set to `-1`. A negative count also produces an empty row, and a count
  above the ordinary window length is clipped to that length.
- Supply all tensors on the same NPU. Caller-provided output buffers must be
  contiguous with the shapes and dtypes above. The function writes into and
  returns those buffers, allowing graph replay callers to retain their
  addresses.

## Origin and Differences

- Origin. vLLM Ascend's existing causal window index builder for the A5
  packed-cache path.
- Differences. The replay path clips the ordinary count before generating the
  indices. This preserves the newest readable KV positions at the start of
  each row. When the optional counts are omitted, the kernel uses the ordinary
  causal window.

## Test Cases

The A5 single-card test checks exact INT32 indices and counts at the model's
128-token window size, including mixed requests, the replay boundary, padded
rows, and reuse of output buffers. It also checks empty input, non-contiguous
visible counts, negative and oversized counts, and a non-power-of-two window.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_build_window_indices.py
```

## Example

With `positions = [11, 0]`, `window_size = 4`, and `visible_lengths = [2]`,
the first row is `[10, 11, -1, -1]` with count 2. The second row is padding
and contains four `-1` entries with count 0. The ordinary window for position
11 would contain `[8, 9, 10, 11]`.
