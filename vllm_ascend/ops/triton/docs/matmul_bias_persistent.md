# matmul_bias_persistent

## Description

- **Location**: `vllm_ascend/ops/triton/batch_invariant/matmul.py` — `matmul_bias_persistent_kernel`, 1:1 wrapper `matmul_persistent`, convenience wrappers `mm_batch_invariant` / `bmm_batch_invariant` / `addmm_batch_invariant` / `matmul_batch_invariant`
- **Function**: Batch-invariant GEMM `x @ y (+ bias)` with fp32 accumulation and TF32 disabled, so the result is deterministic and independent of batch composition. It is one of the four Triton kernels of the batch-invariant family adapted from vllm for NPU. In batch-invariant mode the vllm-ascend facade (`vllm_ascend/batch_invariant.py::enable_batch_invariant_mode`) registers it as an `aten` implementation on the NPU dispatch key: `aten::addmm` and `aten::bmm` whenever Triton is available, plus `aten::mm` and `aten::matmul` when the AscendC `batch_invariant_ops` package is absent (the AscendC ops take priority when present). Mode activation: `--additional-config '{"enable_batch_invariant": true}'` (sets `VLLM_BATCH_INVARIANT=1`), initialized per rank in `vllm_ascend/worker/worker.py` (`init_batch_invariance`).
- **Formula** (per output element, computed in fp32):
    - `out[m, n] = Σ_k x[m, k] · y[k, n] + bias[n]` (bias term only when `bias` is given)
    - Inputs are cast to fp32 on load; `tl.dot(..., allow_tf32=False)` accumulates into an fp32 register tile; the result is stored in the input dtype.
- **Algorithm flow** (one output tile per program):
  1. Grid `(ceil(M / BLOCK_M), ceil(N / BLOCK_N))` with fixed `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`.
  2. Per output tile: loop over `K` in `BLOCK_K` chunks; load the `x` row-block and `y` column-block with boundary masks (`other=0.0`), cast to fp32, accumulate `tl.dot` into `acc`.
  3. If `has_bias`: load the `bias[n]` slice (row-broadcast) and add it to `acc`.
  4. Store `acc` cast to the input dtype with an `(m < M) & (n < N)` mask, so arbitrary (non-aligned) `M`/`N`/`K` are handled without padding.
- **Supported modes**: Atlas A2 and Atlas A3. Used by every model layer that goes through `torch.mm` / `torch.addmm` / `torch.bmm` / `torch.matmul` while batch-invariant mode is enabled (e.g. linear layers, attention score/value projections); works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | LHS matrix `[M, K]`; made contiguous by the wrapper | fp16 / bf16 / fp32 | ND |
| `y` | Input | RHS matrix `[K, N]`; made contiguous by the wrapper | fp16 / bf16 / fp32 | ND |
| `bias` | Input | Optional addend `[N]`; pass `None` to skip (kernel flag `has_bias=False`) | fp16 / bf16 / fp32 | ND |
| `output` | Output | `x @ y (+ bias)` `[M, N]`, same dtype as `x` | fp16 / bf16 / fp32 | ND |

## Constraints

- `x` and `y` must be 2D with `x.shape[1] == y.shape[0]`; `bias` must be 1D with `bias.shape[0] == N` (asserted by the wrapper).
- Only 2D × 2D is implemented in the kernel; batched/higher-rank cases are handled by the convenience wrappers (`bmm` loops over the batch dim, `matmul` reshapes 3D/4D combos, `addmm` falls back to `matmul + scale` unless `alpha == beta == 1` with a 1D `input`).
- Tiling is fixed at compile time (`BLOCK_M/N/K = 128/128/64`); non-aligned shapes are masked, not padded.
- Accumulation is always fp32 with `allow_tf32=False`; the output dtype equals the input dtype.
- Only for inference on NPU with batch-invariant mode enabled.

## Origin and Differences

- **Origin**: Adapted from vllm's `model_executor/layers/batch_invariant.py` (`matmul_kernel_persistent` / `matmul_persistent`); the NPU port first landed with the batch-invariant framework in #5517.
- **Differences**:
    - NPU adaptation for performance: replaces upstream's persistent 1D grid with `_compute_pid` L2-grouped tile ordering and autotuned block sizes by a plain 2D block grid with fixed `128/128/64` tiles and boundary masks, sized for the NPU vector cores;
    - Modified for a specific vllm-ascend logic or different input parameters: adds the fused optional bias (`has_bias` constexpr + broadcast add) so `aten::addmm` can be served by a single launch, and forces fp32 loads plus `allow_tf32=False` for strict batch invariance.

## Test Cases

The test guards the 1:1 wrapper against an fp32 CPU golden, across fp16/bf16, `bias` on/off, and shapes covering a single row, non-aligned sizes, and multi-block tiling (`(m, k, n) = (1, 64, 128)`, `(17, 65, 129)`, `(129, 257, 131)`), with unified tolerances (`rtol=2e-3, atol=2e-2` for fp16; `2e-2, 5e-2` for bf16). Wrapper-level cases for `mm`/`bmm`/`addmm`/`matmul` shape combinations and the `out=` parameter are covered by the sibling tests in the same file.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_invariant_ops.py -k matmul_bias_persistent
```
