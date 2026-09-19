# solve_tril

## Description

- **Function**: Computes `inverse(I + A)` independently for each strictly lower-triangular chunk matrix produced by `chunk_scaled_dot_kkt_fwd`. The result is consumed by WY operators such as `recompute_w_u_fwd`.
- **Formula**: Inverts a unit lower-triangular matrix by solving 16-by-16 diagonal blocks and merging them recursively:
- Input `A`: `[B, T, H, BT]`, where `BT` is 16, 32, or 64
- Base inverse: `A_inv = inverse(I + A)`
- Diagonal blocks: `A_inv_11 = inverse(L_11)` and `A_inv_22 = inverse(L_22)`
- Lower off-diagonal block: `A_inv_21 = -A_inv_22 * A_21 * A_inv_11`
- Upper off-diagonal block: zero because `I + A` is lower triangular
- Output `A_inv`: `[B, T, H, BT]`, containing the valid unit lower-triangular inverse of each chunk
- **Algorithm flow**:
  1. Solve every 16-by-16 diagonal block with forward substitution.
  2. Return these blocks directly when `BT=16`.
  3. For `BT=32` or `BT=64`, recursively merge the diagonal inverses and compute the lower off-diagonal blocks with matrix multiplications.
  4. Store the valid unit lower-triangular inverse for each chunk and head.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950 (Triton kernel); block sizes 16, 32, and 64; fixed-length and variable-length sequences; eager and graph-capture modes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `A` | Input | Strictly lower-triangular chunk matrix `[B, T, H, BT]` | fp32 / fp16 / bf16 | ND |
| `cu_seqlens` | Input | Cumulative sequence lengths `[N + 1]` | int32 / int64 | ND |
| `chunk_indices_large_block` | Input | Variable-length indices for the 16-by-16 solve stage; generated when omitted | int32 / int64 | ND |
| `chunk_indices_bt` | Input | Variable-length indices for the `BT` merge stage; generated when omitted | int32 / int64 | ND |
| `output_dtype` | Input (attribute) | Requested output dtype; default `torch.float` | torch dtype | scalar |
| `A_inv` | Output | Chunk-local inverse `(I + A)^-1` with shape `[B, T, H, BT]` | specified by `output_dtype` | ND |

## Constraints

- The last dimension `BT` of `A` must be 16, 32, or 64.
- `A` must be strictly lower triangular inside each valid chunk.
- Input tensors must be contiguous in the last dimension.
- Variable-length sequences are strictly isolated from one another.
- Precomputed chunk indices must match `cu_seqlens` and their respective chunk sizes.
- Only the valid lower-triangular region is defined for a tail chunk shorter than `BT`.

## Origin and Differences

- **Origin**: Based on the triangular solve from the flash-linear-attention project (MIT license; see the source-file header).
- **Differences**:
    - Adapted to Ascend NPU with a 16-by-16 base solve and explicit 32-by-32/64-by-64 merge kernels.
    - Uses Ascend-compatible slice primitives, masked tail handling, and variable-length chunk indices.

## Test Cases

The test compares valid lower-triangular results with `torch.linalg.inv(I + A)` for `BT=16/32/64`, fixed-length and variable-length inputs, tail chunks, multiple heads, and fp32/bf16 output.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_solve_tril.py
```
