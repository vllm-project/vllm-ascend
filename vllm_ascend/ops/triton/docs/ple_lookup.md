# BF16 PLE host lookup

## Description

- **Function**: Gather selected Qwen3.8-Flash-Next PLE embedding rows stored
  in CANN-registered host memory directly into a caller-provided NPU BF16
  output. The quantized counterpart is documented separately in
  [ple_lookup_quantized.md](ple_lookup_quantized.md).
- **Formula**: For an owned global ID `i`, let row `r = i - shard_start` when
  `global_rows` is false, or `r = i` when it is true. For each output column
  `c < width`, `output[i, c] = table[r, c]`; non-owned IDs and columns outside
  `width` produce zero.
- **Algorithm flow** (one program per requested row, independently): load the
  ID, check shard ownership, select the host allocation chunk through the
  device pointer table, load the row from registered host memory, and store it
  to the output. Columns outside the width are masked.
- **Supported modes**: Atlas A2, Atlas A3, and 950PR&950DT (A5). The
  shared-table `global_rows=True` path added by this change was exercised on
  A2; A5 hardware was not run in this PR.

## Parameters

All parameters are required except `global_rows`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `ptrs` | Input | Device addresses of registered host chunks | NPU INT64 | Contiguous 1-D |
| `ids` | Input | Global embedding IDs to gather | NPU INT32/INT64 | Any shape; flattened and copied to contiguous when needed |
| `output` | Output | BF16 rows written in place | NPU BF16 | `(*ids.shape, width)`; last dimension is `width` |
| `start`, `end` | Attribute | Inclusive/exclusive global ID bounds owned by the caller | Python int | `0 <= start < end` |
| `chunk` | Attribute | Rows represented by one pointer-table entry | Python int | Positive |
| `global_rows` | Attribute | Interpret IDs as absolute shared-table rows instead of shard-local rows | Python bool | Keyword-only; default `False` |

When `global_rows=False`, the pointer table must cover `end - start` rows and
IDs are shard-local indices offset by `start`. When `global_rows=True`, IDs
address the shared table directly, so the pointer table must cover rows up to
`end - 1`. Both modes keep the same ownership check.

## Constraints

- The table is BF16; the output must be BF16 with the same row width.
- All device tensors must be on the same NPU. Host allocations must remain
  registered and alive until all eager and graph work completes.
- The pointer table must cover every row reachable for the selected mode and
  `chunk` must match the host allocation layout.
- IDs outside `[start, end)`, including negative IDs, are zeroed without an
  error because other ranks may own them.
- The output must not alias the pointer table, IDs or host storage. Only the
  output is modified. Empty IDs perform no launch after validation.
- NPU graph replay is supported; graphs must be torn down before closing host
  buffers.

`GLOBAL_ROWS` is a `tl.constexpr`, so the two modes compile as two variants.
The value is fixed per embedding instance at construction time, not per
request, so no token- or batch-shaped variants are introduced. No dedicated
warmup registration is added; the selected variant compiles on first use.

## Origin and Differences

- **Origin**: Derived from the existing Engram host-UVA gather in
  `ops/triton/engram_lookup.py`; the production caller is
  `AscendPLEHostEmbedding._lookup` in `models/qwen4_exp/ple_embedding.py`.
- **Differences**: Adds `global_rows` so a shared host table can be indexed by
  absolute row while a private host table keeps shard-local rows. Both modes
  perform the same ownership check and zeroing. This is a model-support
  feature; no throughput improvement is claimed.

## Test Cases

The device tests compare BF16 output exactly (`rtol=0`, `atol=0`) against a
PyTorch reference. Cases cover private and shared row addressing, shard and
chunk boundaries, non-owned and negative IDs, empty IDs, non-contiguous ID
views, and NPU graph replay.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_ple_host_lookup.py
```

Model-level dataset accuracy and latency/throughput must be reported separately
in the PR, with software revisions, checkpoint, hardware, parallelism, offload
settings, graph/MTP settings and evaluation parameters.

## Example

```python
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup import lookup

lookup(shared_ptrs, ids, output, vocab_start, vocab_end, rows_per_mapping, global_rows=True)
```
