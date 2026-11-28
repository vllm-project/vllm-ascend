# chunk_scaled_dot_kkt_fwd

## Description

- **Function**: Computes the strictly lower-triangular WY transformation matrix `A = beta * (K @ K.T)` within each Gated Delta Rule chunk, with optional cumulative gating. The result is consumed by `solve_tril`.
- **Formula**: Computes a gated, scaled key Gram matrix and applies a strict lower-triangular mask:
- Input `k`: `[B, T, Hg, K]`; input `beta`: `[B, T, H]`; optional input `g_cumsum`: `[B, T, H]`
- Key correlation: `C[i, j, h] = sum_d(k[i, h_g, d] * k[j, h_g, d])`
- Lower-triangular value: `A[i, j, h] = beta[i, h] * C[i, j, h] * exp(g[i, h] - g[j, h])` when `i > j`
- Masked value: `A[i, j, h] = 0` when `i <= j`
- Output `A`: `[B, T, H, BT]`, where `BT = chunk_size`
- Head mapping: `h_g = floor(h / (H / Hg))`; gating is omitted when `g_cumsum=None`; `safe_exp` suppresses positive exponents to avoid overflow
- **Algorithm flow**:
  1. Partition each sequence into chunks of `chunk_size` and map output heads to KV heads.
  2. Load key tiles and compute the chunk-local key Gram matrix.
  3. Apply `beta`, the optional cumulative-gate difference, and the strict lower-triangular mask.
  4. Store the result in `[B, T, H, BT]` layout.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel); fixed-length and variable-length sequences; GQA/MQA; eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `k` | Input | Key tensor `[B, T, Hg, K]` | fp32 / fp16 / bf16 | ND |
| `beta` | Input | Per-token scale tensor `[B, T, H]` | fp32 / fp16 / bf16 | ND |
| `g_cumsum` | Input | Cumulative gate `[B, T, H]`; `None` disables gating | fp32 / fp16 / bf16 | ND |
| `cu_seqlens` | Input | Cumulative sequence lengths `[N + 1]` | int32 / int64 | ND |
| `chunk_indices` | Input | Variable-length chunk indices `[NT, 2]`; generated when omitted | int32 / int64 | ND |
| `chunk_size` | Input (attribute) | Number of tokens per chunk; default 64 | int32 | scalar |
| `output_dtype` | Input (attribute) | Requested output dtype; default `torch.float32` | torch dtype | scalar |
| `A` | Output | Strictly lower-triangular chunk matrix `[B, T, H, BT]` | specified by `output_dtype` | ND |

## Constraints

- `H` must be divisible by `Hg`.
- The last dimension `BT` of `A` equals `chunk_size`.
- The `(B, T)` dimensions of `k`, `beta`, and `g_cumsum` must match.
- `g_cumsum` should be the log-space cumulative gate produced by `chunk_local_cumsum` and is normally non-increasing.
- Variable-length sequences are strictly isolated from one another.
- Rows and columns beyond the valid length of a tail chunk are undefined and must not be read.

## Origin and Differences

- **Origin**: Based on the `chunk_scaled_dot_kkt_fwd` implementation from the flash-linear-attention project (MIT license; see the source-file header).
- **Differences**:
    - Adapted to Ascend NPU Triton primitives and device execution characteristics.
    - Uses Ascend-compatible masking, `safe_exp`, and variable-length chunk indexing.

## Test Cases

The test checks fixed-length and variable-length inputs, GQA head mappings, gated and ungated paths, tail chunks, and supported floating-point dtypes against a PyTorch reference.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_scaled_dot_kkt_fwd.py
```
