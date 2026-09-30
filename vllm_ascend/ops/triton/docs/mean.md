# mean

## Description

- **Location**: `vllm_ascend/ops/triton/batch_invariant/mean.py` — `mean_kernel`, 1:1 wrapper `mean_dim`, convenience wrapper `mean_batch_invariant`
- **Function**: Batch-invariant mean along a single dimension: reduces the input viewed as `(M, N, K)` over the middle (reduced) dimension `N` with a running scalar accumulator. Part of the batch-invariant Triton family adapted from vllm for NPU.
- **Formula** (per output element, computed independently):
    - `out[m, k] = (1 / N) · Σ_n x[m, n, k]` where the input is reshaped to `(M, N, K)` with `N` the reduced dimension (`M`/`K` are the products of the dims before/after it).
- **Algorithm flow** (one program per output element):
  1. Grid `(M * K,)`; program `pid` decodes `(m_idx, k_idx) = (pid // K, pid % K)` with a bounds check.
  2. Strided loop over the reduction dimension in `BLOCK_SIZE = 1024` chunks with a tail mask (`other=0.0`), accumulating `tl.sum` of each chunk into a running scalar.
  3. Divide by `N` and store the single output element.
- **Supported modes**: Atlas A2 and Atlas A3. Intended for the batch-invariant reduction path.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input_` | Input | Tensor of arbitrary rank; the dim before/after `dim` are folded into `M`/`K` | fp16 / bf16 / fp32 (integers are promoted to fp32 when `dtype=None`) | ND |
| `dim` | Input | Single dimension to reduce; negative values normalized by the wrapper | int32 | scalar |
| `keepdim` | Input | Whether the reduced dim is kept as size 1 (default `False`) | bool | scalar |
| `dtype` | Input | Compute/output dtype (default `torch.float16`); the input is cast to it first — unlike `torch.mean`, this changes the reduction dtype, not only the output dtype | torch.dtype | attribute |
| `output` | Output | Mean over `dim`, shape `input.shape` with `dim` removed (or size 1 if `keepdim`) | same as `dtype` | ND |

## Constraints

- Single-dimension reduction only; `mean_batch_invariant` handles multi-dim `dim` lists by falling back to `torch.sum(..., dtype=fp32) / n_elems`, and asserts `dtype is None or fp32`.
- The `dtype` cast happens before the reduction (`input_.to(dtype)`), so a `None`-default fp32 reduction requires passing `dtype=torch.float32` explicitly (the wrapper default is fp16).
- The convenience wrapper's `dtype` assert applies to its own path only; the 1-dim case delegates to `mean_dim` with the wrapper's own default dtype.
- `BLOCK_SIZE = 1024` is fixed; arbitrary `N` handled by masking.
- Only for inference on NPU.

## Origin and Differences

- **Origin**: Adapted from vllm's `model_executor/layers/batch_invariant.py` (`mean_kernel` / `mean_dim`, registered upstream as the `aten::mean.dim` batch-invariant implementation); the kernel body and `(M, N, K)` view are carried over essentially unchanged (#5517).
- **Differences**:
    - NPU adaptation for performance: none required in the kernel body — it is a direct port; the grid `(M*K,)` of one-program-per-output-element maps naturally onto the vector cores;
    - Modified for a specific vllm-ascend logic or different input parameters: the wrapper default `dtype` is `torch.float16` instead of upstream's `None` (input dtype), and upstream's `aten::mean.dim` registration is not wired on the NPU facade (reductions use the AscendC `npu_reduce_sum_batch_invariant` op instead).

## Test Cases

The test guards the 1:1 wrapper against `torch.mean` on fp32 CPU, across fp16/bf16/fp32, dimensions exercising multi-block reduction tails, negative `dim`, and `keepdim` (`(3, 1031, 17)` dim 1, `(5, 7, 2051)` dim -1 keepdim, `(2051, 3)` dim 0 keepdim), with unified tolerances (`rtol=2e-3, atol=2e-2` for fp16; `2e-2, 5e-2` for bf16; `1e-4, 1e-4` for fp32). The sibling `test_mean_batch_invariant_multiple_dims` covers the convenience wrapper's multi-dim fallback.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_invariant_ops.py -k mean_kernel
```
