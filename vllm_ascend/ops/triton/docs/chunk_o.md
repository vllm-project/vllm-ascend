# chunk_fwd_o

## Description

- **Function**: Computes the output of each chunk in the Gated Delta Rule forward pass by combining the inter-chunk hidden-state contribution (`q @ h`) with the intra-chunk attention contribution (`A_intra @ v`), with optional gating.
- **Formula**: Combines the inter-chunk hidden-state contribution with the causal intra-chunk attention contribution:
- Input `q`, `k`: `[B, T, Hg, K]`; input `v`: `[B, T, H, V]`; hidden state `h`: `[B * NT, H, K, V]`
- Query-key correlation: `C[i, j, h] = sum_d(q[i, h_g, d] * k[j, h_g, d])`
- Intra-chunk matrix: `A_intra[i, j, h] = indicator(i >= j) * C[i, j, h] * exp(g[i, h] - g[j, h])`
- Inter-chunk contribution: `o_inter[i, h, d_v] = scale * exp(g[i, h]) * sum_d(q[i, h_g, d] * hidden[c, h, d, d_v])`
- Intra-chunk contribution: `o_intra[i, h, d_v] = scale * sum_j(A_intra[i, j, h] * v[j, h, d_v])`
- Output `o`: `[B, T, H, V]`, where `o = o_inter + o_intra`
- Head mapping: `h_g = floor(h / (H / Hg))`; `scale = K**-0.5` when `scale=None`; gating is omitted when `g=None`
- **Algorithm flow**:
  1. Map each output head to the corresponding query/key head for GQA or MQA.
  2. Load the hidden state accumulated before the current chunk and compute the inter-chunk contribution.
  3. Compute the causal query-key correlation matrix within the chunk and apply the optional cumulative gate.
  4. Multiply the intra-chunk correlation by `v`, combine both contributions, and store `o`.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel); fixed-length and variable-length sequences; GQA/MQA; eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q` | Input | Query tensor `[B, T, Hg, K]` | fp32 / fp16 / bf16 | ND |
| `k` | Input | Key tensor `[B, T, Hg, K]` | fp32 / fp16 / bf16 | ND |
| `v` | Input | Value tensor `[B, T, H, V]` | fp32 / fp16 / bf16 | ND |
| `h` | Input | Inter-chunk hidden state `[B * NT, H, K, V]` | fp32 / fp16 / bf16 | ND |
| `g` | Input | Cumulative gate `[B, T, H]`; `None` disables gating | fp32 / fp16 / bf16 | ND |
| `scale` | Input (attribute) | Scale factor; defaults to `K**-0.5` | fp32 | scalar |
| `cu_seqlens` | Input | Cumulative sequence lengths `[N + 1]` | int32 / int64 | ND |
| `chunk_size` | Input (attribute) | Number of tokens per chunk; default 64 | int32 | scalar |
| `chunk_offsets` | Input | Per-sequence chunk offsets `[N + 1]`; generated when omitted | int32 / int64 | ND |
| `o` | Output | Chunk output `[B, T, H, V]` | same as `v` | ND |

## Constraints

- `H` must be divisible by `Hg`.
- The first dimension of `h` must match the total number of chunks; variable-length mode must use matching `chunk_offsets`.
- The `(B, T)` dimensions of `q`, `k`, `v`, and `g` must match.
- `g` should be the log-space cumulative gate produced by `chunk_local_cumsum`.
- Input tensors must be contiguous in the last dimension.
- The current default and recommended `chunk_size` is 64.

## Origin and Differences

- **Origin**: Based on the `chunk_fwd_o` implementation from the flash-linear-attention project (MIT license; see the source-file header).
- **Differences**:
    - Adapted to Ascend NPU with Triton and NPU-specific tiling and device-property selection.
    - Supports vLLM Ascend variable-length metadata and graph capture while preserving the GQA/MQA head mapping.

## Test Cases

The test compares the Triton result with a PyTorch reference for fixed-length and variable-length inputs, multiple head mappings, and supported floating-point dtypes.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_fwd_o.py
```
