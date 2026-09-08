# rms_norm

## Description

- **Location**: `vllm_ascend/ops/triton/batch_invariant/rmsnorm.py` — `_rms_norm_kernel`, 1:1 wrapper `rms_norm`, convenience wrapper `rms_norm_batch_invariant`.
- **Function**: Batch-invariant RMS normalization along the last dimension with all arithmetic in fp32. Part of the batch-invariant Triton family adapted from vllm for NPU.
- **Formula** (per row, computed in fp32):
    - `inv_rms = 1 / sqrt(mean(x²) + eps)` with `mean(x²) = (1/n_cols) · Σ_j x[j]²`
    - `y[j] = x[j] · inv_rms · weight[j]`, stored in the input dtype
- **Algorithm flow** (rows share programs):
  1. Input is flattened to 2D `[n_rows, n_cols]`; grid is `min(n_rows, num_vectorcore)` programs.
  2. Each program owns `ceil(n_rows / n_programs)` consecutive rows; per row it makes two passes over the columns in `BLOCK_SIZE = 1024` chunks with tail masks: pass 1 accumulates the sum of squares in fp32, pass 2 normalizes, scales by `weight`, and stores in the input dtype.
- **Supported modes**: Atlas A2 and Atlas A3. Intended for the batch-invariant normalization path.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input_` | Input | Activations `[..., hidden_size]`; leading dims are flattened to rows, made contiguous by the wrapper | fp16 / bf16 / fp32 | ND |
| `weight` | Input | Per-channel scale `[hidden_size]` (**required**, unlike upstream's optional weight); made contiguous by the wrapper | fp16 / bf16 / fp32 | ND |
| `eps` | Input | Numerical-stability constant added to the mean square (default `1e-6`) | fp32 | scalar |
| `output` | Output | Normalized tensor with the input's original shape and dtype | fp16 / bf16 / fp32 | ND |

## Constraints

- `weight` must be 1D with `weight.shape[0] == input_.shape[-1]` (asserted).
- `BLOCK_SIZE = 1024` is fixed; arbitrary `hidden_size` handled by masking (tested at 1025 and 4096).
- All arithmetic is fp32 (loads cast, stores cast back); the output dtype equals the input dtype.
- The grid is capped at the vector-core count with rows block-scheduled per program, so program count stays independent of the token count.
- Only for inference on NPU.

## Origin and Differences

- **Origin**: Adapted from vllm's `model_executor/layers/batch_invariant.py` (`_rms_norm_kernel` / `rms_norm_batch_invariant`); the three-pass math (sum of squares → inv_rms → normalize+scale) is carried over unchanged (#5517).
- **Differences**:
    - NPU adaptation for performance: replaces upstream's one-row-per-program grid (`grid = (n_rows,)`) with a capped grid of `min(n_rows, num_vectorcore)` programs, each owning a contiguous block of rows — fewer programs with more work each on the NPU vector cores; fixed `BLOCK_SIZE=1024` instead of upstream's autotuned meta-parameters;
    - Modified for a specific vllm-ascend logic or different input parameters: `weight` is mandatory (upstream has a `HAS_WEIGHT` constexpr for the optional-weight case), and upstream's `aten` registration is not wired on the NPU facade (the fused add+rmsnorm path uses `torch_npu.npu_rms_norm` instead).

## Test Cases

The test guards the 1:1 wrapper against an fp32 CPU golden (`x · rsqrt(mean(x²) + eps) · w`) across fp16/bf16/fp32, a non-power-of-two hidden size (1025) and a multi-program/multi-block shape (2×65×4096), `eps = 1e-6 / 1e-5`, with unified tolerances (`rtol=2e-3, atol=2e-2` for fp16; `2e-2, 5e-2` for bf16; `1e-4, 1e-4` for fp32).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_invariant_ops.py -k rms_norm_kernel
```
