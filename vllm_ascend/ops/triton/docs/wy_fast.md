# recompute_w_u_fwd

## Description

- **Function**: Recomputes the intermediate WY-representation tensors `w` and `u` during the Gated Delta Rule chunk scan from the solved transformation matrix `A`, key/value tensors, scale factor `beta`, and cumulative gate `g_cumsum`. The resulting `u` is the updated value used by subsequent computations.
- **Formula**: Multiplies the solved chunk matrix by gated, scaled keys and scaled values:
- Input `k`: `[B, T, Hg, K]`; input `v`: `[B, T, H, V]`; input `A`: `[B, T, H, BT]`
- Scaled key: `k_scaled[j, h, d] = k[j, h_g, d] * beta[j, h] * exp(g_cumsum[j, h])`
- Scaled value: `v_scaled[j, h, d_v] = v[j, h, d_v] * beta[j, h]`
- Weighted key: `w[i, h, d] = sum_j(A[i, j, h] * k_scaled[j, h, d])`
- Weighted value: `u[i, h, d_v] = sum_j(A[i, j, h] * v_scaled[j, h, d_v])`
- Outputs `w`, `u`: `[B, T, H, K]` and `[B, T, H, V]`
- Head mapping: `h_g = floor(h / (H / Hg))`; `A` is the unit lower-triangular inverse produced by `solve_tril`
- **Algorithm flow**:
  1. Partition each sequence into chunks and map output heads to KV heads.
  2. Scale keys by `beta * exp(g_cumsum)` and values by `beta`.
  3. Multiply the solved chunk matrix `A` by the scaled keys and values.
  4. Store the weighted key `w` and updated value `u` in their original sequence layouts.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel); fixed-length and variable-length sequences; GQA/MQA; eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `k` | Input | Key tensor `[B, T, Hg, K]` | fp32 / fp16 / bf16 | ND |
| `v` | Input | Value tensor `[B, T, H, V]` | fp32 / fp16 / bf16 | ND |
| `beta` | Input | Per-token scale tensor `[B, T, H]` | fp32 / fp16 / bf16 | ND |
| `g_cumsum` | Input | Cumulative gate `[B, T, H]` from `chunk_local_cumsum` | fp32 / fp16 / bf16 | ND |
| `A` | Input | Unit lower-triangular inverse `[B, T, H, BT]` from `solve_tril` | fp32 / fp16 / bf16 | ND |
| `cu_seqlens` | Input | Cumulative sequence lengths `[N + 1]` | int32 / int64 | ND |
| `chunk_indices` | Input | Variable-length chunk indices `[NT, 2]`; generated when omitted | int32 / int64 | ND |
| `w` | Output | Weighted key `[B, T, H, K]` | same as `k` | ND |
| `u` | Output | Weighted value `[B, T, H, V]` | same as `v` | ND |

## Constraints

- `H` must be divisible by `Hg`.
- The last dimension `BT` of `A` determines the chunk size.
- The `(B, T)` dimensions of `k`, `v`, `beta`, `g_cumsum`, and `A` must match.
- Fixed-length mode currently supports only `B=1`; represent multiple sequences as a flattened variable-length input with `B=1` and `cu_seqlens`.
- `g_cumsum` should be the log-space cumulative gate produced by `chunk_local_cumsum`.
- Input tensors must be contiguous in the last dimension.
- Variable-length sequences are strictly isolated from one another.

## Origin and Differences

- **Origin**: Based on the WY recomputation implementation from the flash-linear-attention project (MIT license; see the source-file header).
- **Differences**:
    - Adapted to Ascend NPU Triton primitives and its matrix/vector execution characteristics.
    - Supports vLLM Ascend GQA/MQA mapping, variable-length chunk metadata, and graph capture.

## Test Cases

The test compares `w` and `u` with a PyTorch reference for fixed-length and variable-length inputs, GQA/MQA head mappings, multiple sequence lengths, and fp32/bf16 data.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_recompute_w_u_fwd.py
```
