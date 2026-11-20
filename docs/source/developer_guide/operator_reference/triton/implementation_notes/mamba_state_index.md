# Mamba State Gather and Scatter

## Description

- **Function**: `gather_ssm_states` gathers selected physical SSM cache rows
  into a contiguous tensor. `scatter_ssm_states_` writes updated rows back
  into the original cache allocation in place.
- **Formula**: Let `S` have shape `[N, *inner_dims]`, let `I` contain `B`
  physical row IDs, and let `M` be the initial-state mask. For each request
  `b`, gather produces `O[b] = cast(S[I[b]], output_dtype)` when `M[b]` is
  true, and a zero row otherwise. Scatter performs
  `S[I[b]] = cast(source[b], S.dtype)`.
- **Algorithm flow**: Flatten the inner dimensions into a row of `R`
  elements. Launch a two-dimensional grid
  `[ceil(R / STATE_IO_BLOCK_SIZE), B]`, with `STATE_IO_BLOCK_SIZE = 1024`.
  Each program copies one tile of one selected row. Address calculation uses
  the physical row stride, and a mask protects the final partial tile.
  Gather also masks cache reads for requests without initial state.
- **Supported modes**: Ascend NPU execution through the installed Triton
  backend. Hardware and graph support depend on that backend. The numerical
  test below runs directly on NPU; it does not independently certify graph
  capture or every hardware generation.

## Parameters

`output_dtype` is optional; all other parameters are required for their
respective function. `N` is the number of physical state rows, and `B` is the
number of selected rows.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state` | Gather input; scatter input/output | Cache with shape `[N, *inner_dims]` | Floating-point tensor; numerical cases cover FP32 state | ND, contiguous inner dimensions, optionally padded row stride |
| `indices` | Input | Physical row IDs, shape `[B]` | INT32 or INT64 | 1D, strided input supported |
| `has_initial_state` | Gather input | Whether each request reads its cached initial state, shape `[B]` | BOOL | 1D, strided input supported |
| `output_dtype` | Gather attribute | Output dtype; defaults to `state.dtype` | `torch.dtype` or `None` | N/A |
| Gather return value | Output | Selected or zeroed rows, shape `[B, *inner_dims]` | `output_dtype` or `state.dtype` | Contiguous ND |
| `source` | Scatter input | Updated rows, shape `[B, *inner_dims]` | Floating-point tensor; numerical cases cover BF16 source | ND, contiguous inner dimensions |
| Scatter return value | Output | N/A; the function returns `None` and mutates `state` | N/A | N/A |

## Constraints

- `state` must have at least two dimensions and at least one physical row.
  For nonempty launches, inner dimensions must have a positive product.
- Only `state.stride(0)` may contain physical page padding. Every logical
  row must be contiguous, and `state.stride(0)` must be at least `R`.
  Arbitrary inner-dimension strides are not supported.
- `indices` must be one-dimensional INT32 or INT64 and reside on the same
  device as `state`. IDs refer to physical rows rather than token positions.
  IDs may be unsorted, and the indices tensor need not be contiguous.
- Callers must ensure row IDs are in range `[0, N)`. The wrapper does not
  read device index values on the host to validate bounds.
- Gather permits repeated row IDs. For deterministic scatter behavior,
  callers must provide unique destination IDs; duplicate destinations can
  race between programs and are not validated by the wrapper.
- `has_initial_state` must be a BOOL tensor with the same shape and device
  as `indices`. A false entry produces a zero output row without reading the
  corresponding cache row. Gather never modifies the cache.
- `source` must have shape `[B, *inner_dims]`, use the same device as `state`,
  and have contiguous inner dimensions. Its row stride is honored. Callers
  must avoid source/destination aliasing that creates cross-program hazards.
- Gather converts values when storing to `output_dtype`; scatter converts
  values when storing to `state.dtype`. No arithmetic accumulation is used.
- Unselected cache rows and padding are preserved by scatter. Empty
  `indices` produce an empty gather result or a no-op scatter.
- Validation uses tensor metadata rather than reading NPU scalar values.
  Graph-mode integration must still be checked in the model runner with
  its actual shapes and allocation lifetimes.

## Origin and Differences

- **Origin**: New Triton helpers for the Ascend GDN fused-prefill path,
  replacing the combination of advanced indexing, initial-state clearing,
  and indexed write-back.
- **Differences**: Physical row strides are explicit. Only selected logical
  rows are copied, without copying page padding. Gather combines row
  selection, zero initialization, and output dtype conversion in one launch.

## Test Cases

The nightly test contains 18 cases: 16 combinations of dense/padded storage,
INT32/INT64 indices, and four state shapes, plus two invalid-layout checks.
The shapes include `1023`, `1024`, and `1025` elements per row, and the real
GDN state shape `[8, 128, 128]`. This covers tile boundaries, a multi-tile
tail, and a production-sized state row.

The PyTorch reference checks gathered rows, zero initialization, dtype
conversion, selected scatter destinations, an unselected row, and padding.
Comparisons use `rtol=0, atol=0` after applying the same dtype conversions,
because these are data-movement operations. Empty-cache and overlapping-row
validation errors are also checked.

From the repository root, with the Ascend environment initialized:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py
```

## Example

```python
import torch

from vllm_ascend.ops.triton.mamba.state_index import (
    gather_ssm_states,
    scatter_ssm_states_,
)

storage = torch.zeros((4, 20), device="npu", dtype=torch.float32)
state = storage[:, :16].view(4, 2, 8)
indices = torch.tensor([2, 0], device="npu", dtype=torch.int32)
has_initial_state = torch.tensor([True, False], device="npu")
initial = gather_ssm_states(
    state, indices, has_initial_state, output_dtype=torch.bfloat16
)
scatter_ssm_states_(state, indices, initial)
```
