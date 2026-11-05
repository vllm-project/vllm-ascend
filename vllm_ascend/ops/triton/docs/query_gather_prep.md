# prep_query_head_major

## Description

- **Function**: Assemble the two DCP query fragments directly into contiguous `[H,T,N+R]` communication order.
- **Formula**: `out[h,t,:N] = q_nope[t,h,:]`; `out[h,t,N:] = q_rope[t,h,:]`.
- **Algorithm flow**: Specialize fixed head/feature dimensions and feature strides while retaining dynamic token counts and other strides. Traverse token tiles within each head with a bounded one-dimensional grid; head selection uses scalar arithmetic. On A5, batch eight rows when at least eight rows per vector core exist; otherwise process one row per iteration. Row/feature masks protect tail accesses while preserving affine DMA addresses. Omit zero-width fragments at compile time; return an empty tensor without launching when the output is empty.
- **Supported modes**: NPU inference. A5 strided-view and graph cases are device-tested; A2/A3 retain one-row tiles but are not hardware-validated here.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `ql_nope` | Input | Query content `[T,H,N]`, including strided views | Same dtype as `q_pe` | ND |
| `q_pe` | Input | Query positional fragment `[T,H,R]`; R may be zero | Same dtype as `ql_nope` | ND |
| Return | Output | Contiguous `[H,T,N+R]` payload | Input dtype | ND |

## Constraints

- Inputs have matching token/head dimensions and dtype, and belong to the same device.
- Source strides, including feature strides, are used explicitly. Positive and zero strides are supported. Neither fragment needs a contiguous copy.
- Legal views and zero-width fragments stay on Triton. Kernel/import failures propagate; no `None` or Torch fallback is provided.
- Tail masks protect stores and loads. Every enabled access belongs to the source view or destination allocation; disabled lanes perform no memory access.

## Origin and Differences

- **Origin**: Existing DCP query-gather preparation kernel.
- **Differences**: Removes the conservative `numel()` storage proof and caller exception fallback. Adds feature strides, zero-width fragments and A5 row batching. Head-aligned tiles preserve affine strides; extra address clamping would inhibit DMA lowering. Physical PCP/TP head ownership and communication ordering are unchanged.

## Test Cases

CPU routing tests use a test-only reference and verify error propagation. Device tests cover split views, feature stride two, non-power-of-two widths, zero-width fragments, empty inputs and changed graph inputs. CPU address simulation is supplementary and does not validate Triton lowering.

```bash
pytest -q tests/e2e/pull_request/one_card/test_dcp_pre_post.py -k query_prep
python benchmarks/dcp_triton_ops.py --only query_gather_prep --output query-prep.json
```
