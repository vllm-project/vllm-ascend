# prep_query_head_major

## Description

- **Function**: Assemble the two DCP query fragments directly into contiguous `[H,T,N+R]` communication order.
- **Formula**: `out[h,t,:N] = q_nope[t,h,:]`; `out[h,t,N:] = q_rope[t,h,:]`.
- **Algorithm flow**: Specialize fixed head/feature dimensions and feature strides while retaining dynamic token counts and other strides. Traverse token tiles within each head with a bounded one-dimensional grid; head selection uses scalar arithmetic. Batch eight rows when at least eight rows per vector core exist; otherwise process one row per iteration. Row/feature masks protect tail accesses while preserving affine DMA addresses. Omit zero-width fragments at compile time; return an empty tensor without launching when the output is empty.
- **Supported modes**: NPU inference, eager and NPUGraph. A3 and A5 use the same kernel and shape-based row tiling. Validation results are recorded in the PR.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `ql_nope` | Input | Query content `[T,H,N]`, including strided views | Same dtype as `q_pe` | ND |
| `q_pe` | Input | Query positional fragment `[T,H,R]`; R may be zero | Same dtype as `ql_nope` | ND |
| Return | Output | Contiguous `[H,T,N+R]` payload | Input dtype | ND |

## Constraints

- Inputs have matching token/head dimensions and dtype, and belong to the same device.
- Head/fragment widths and feature strides are compile-time layout parameters. Token counts and token/head strides remain runtime parameters; token counts do not specialize the kernel.
- Source strides, including feature strides, are used explicitly. Positive and zero strides are supported. Neither fragment needs a contiguous copy.
- Legal views and zero-width fragments stay on Triton. Kernel/import failures propagate; no `None` or Torch fallback is provided.
- Tail masks protect stores and loads. Every enabled access belongs to the source view or destination allocation; disabled lanes perform no memory access.

## Origin and Differences

- **Origin**: Existing DCP query-gather preparation kernel.
- **Differences**: Removes the conservative `numel()` storage proof and caller exception fallback. Supports feature strides, zero-width fragments and shared row batching. Head-aligned tiles preserve affine strides; extra address clamping would inhibit DMA lowering. Physical PCP/TP head ownership and communication ordering are unchanged.

## Test Cases

Nightly NPU tests cover split views, feature stride two, non-power-of-two widths, zero-width fragments, empty inputs and changed graph inputs.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_pre_post.py -k query_prep
```
