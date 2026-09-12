# triton_rms_kernel

## Description

- **Function**: Applies unweighted RMS normalization to query vectors.
- **Formula**: `y = x / sqrt(mean(x^2) + variance_epsilon)` along the last dimension.
- **Algorithm flow** (processed row by row, independently): distribute flattened batch/head rows across vector cores, load row tiles, accumulate fp32 mean squares, scale, and store in the input dtype.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used for query RMS normalization in the DSA/DeepSeek V4 path; supported inside eager and captured model forward.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `hidden_state_ptr` | Input | Flattened query rows `[total_batch, DIM]` | fp16 / bf16 / fp32 | ND |
| `hidden_state_stride_bs` | Attribute | Input row stride | int | scalar |
| `norm_output_ptr` | Output | RMS-normalized rows, same shape as input | same as input | ND |
| `variance_epsilon` | Attribute | Positive numerical-stability epsilon | fp32 | scalar |
| `total_batch` | Attribute | Flattened batch times head count | int | scalar |
| `DIM`, `BLOCK_M` | Attribute | Compile-time row width and rows per tile | int | scalar |

## Constraints

- The public launcher accepts rank-3 contiguous query tensors and requires `DIM <= 2048`.
- `variance_epsilon` must be positive; input and output use the same stride and dtype.
- `BLOCK_M` is a power of two up to 16 selected from rows per vector core.

## Origin and Differences

- **Origin**: Developed in vllm-ascend for the DeepSeek V4 query-normalization path; there is no same-signature upstream vLLM Triton kernel.
- **Differences**:
    - NPU adaptation for performance: flattens batch and head axes and distributes several rows per Ascend vector core.
    - Modified for vllm-ascend logic: omits affine weights and limits the query feature dimension to 2048.

## Test Cases

The NPU test compares against an fp32-accumulating PyTorch reference for shapes `(1,1,128)`, `(2,8,512)`, and `(1,17,2048)`, dtypes fp16/bf16/fp32, and epsilon values `1e-6` and `1e-5`. Tolerances are `(rtol, atol)=(2e-3,2e-2)` for fp16, `(2e-2,5e-2)` for bf16, and `(1e-4,1e-4)` for fp32. It also checks that dimension 2049 is rejected.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rms_norm.py
```
