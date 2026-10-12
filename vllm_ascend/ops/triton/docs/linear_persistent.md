# linear_persistent

## Description

- **Location**: `vllm_ascend/ops/triton/batch_invariant/matmul.py` — `linear_persistent_kernel`, 1:1 wrapper `linear_persistent`, convenience wrapper `linear_batch_invariant`
- **Function**: Batch-invariant `x @ y^T` — the core GEMM of `F.linear`, where the weight `y` is stored as `[N, K]`. Part of the batch-invariant Triton family for NPU. In batch-invariant mode (`--additional-config '{"enable_batch_invariant": true}'` → `VLLM_BATCH_INVARIANT=1`, initialized by `vllm_ascend/worker/worker.py::init_batch_invariance`), the facade registers `linear_batch_invariant` for `aten::linear` on the NPU dispatch key — but only when the AscendC `batch_invariant_ops` package is absent, since AscendC's matmul path gives better performance (upstream vllm instead serves `aten::linear` through its generic `matmul_persistent`).
- **Formula** (per output element, fp32 accumulator):
    - `c[m, n] = Σ_k x[m, k] · y[n, k]` — `y` is used in its natural `[N, K]` layout; the transpose happens in-register via `tl.trans`, so no transposed copy of the weight is materialized.
- **Algorithm flow** (true persistent 1D grid):
  1. `grid_size = num_vectorcore // 2` fixed programs; every program strides over all output tiles: `for block_index in range(pid, NUM_BLOCKS_M * NUM_BLOCKS_N, GRID_SIZE)`.
  2. Per output tile (`BLOCK_M × BLOCK_N`): loop `K` in `BLOCK_K` chunks; load the `a` block `[BLOCK_M, BLOCK_K]` and the `b` block `[BLOCK_N, BLOCK_K]` with boundary masks, transpose `b` in-register, accumulate `tl.dot` into an fp32 tile.
  3. Store the tile cast to the input dtype with `(m < M) & (n < N)` masks.
  4. Block sizes are selected host-side by shape heuristics: `BLOCK_K = 256` (halved for fp32), and `BLOCK_M`/`BLOCK_N` picked from `M`/`N` regimes (e.g. `M < 256` vs. `M ≥ 1024`, `grid_size`-proportional splits) so the tile count roughly saturates the fixed grid.
- **Supported modes**: Atlas A2 and Atlas A3. Serves every `torch.nn.functional.linear` call in batch-invariant mode when the AscendC ops are absent; works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | activations `[M, K]` (the wrapper `linear_batch_invariant` flattens any leading dims to `M`) | fp16 / bf16 / fp32 | ND |
| `y` | Input | linear weight in PyTorch layout `[N, K]` (output features × input features; **not** transposed) | fp16 / bf16 / fp32 | ND |
| `output` | Output | `x @ y^T` `[M, N]`, same dtype as `x` | fp16 / bf16 / fp32 | ND |

## Constraints

- `x` and `y` must be 2D with matching `K` (`x.shape[1] == y.shape[1]`, asserted).
- `M == 0 or N == 0` is legal (empty output; wrapper picks fixed `128/256/256` blocks).
- Accumulation is fp32; the output dtype equals the input dtype; the output is zero-initialized by the wrapper.
- Block sizes are compile-time `constexpr` including `NUM_BLOCKS_M/N` and `GRID_SIZE`, so each distinct heuristic configuration triggers its own Triton compilation.
- Only for inference on NPU with batch-invariant mode enabled.

## Origin and Differences

- **Origin**: Developed in vllm-ascend within the batch-invariant framework (#5517); there is no direct upstream counterpart — upstream vllm's `linear_batch_invariant` computes `x @ weight.T` through the generic `matmul_persistent` GEMM. The file carries the "Adapt from `vllm/model_executor/layers/batch_invariant.py`" header.
- **Differences**:
    - NPU adaptation for performance: dedicated `x @ y^T` kernel with in-register `tl.trans` (avoids materializing `y^T`) and a true persistent 1D grid pinned to half the vector-core count, each program looping over multiple output tiles — program count is independent of the problem size;
    - Modified for a specific vllm-ascend logic or different input parameters: host-side block-size heuristics keyed on `M`/`N`/`grid_size` (several shape regimes) instead of upstream's autotuner; bias is **not** fused — `linear_batch_invariant` adds it as a separate elementwise op after the GEMM.

## Test Cases

The test guards the 1:1 wrapper against an fp32 CPU golden across fp16/bf16/fp32 and shapes exercising the small, medium, and large-`M` heuristic regimes (`(m, k, n) = (7, 63, 129)`, `(257, 257, 131)`, `(1025, 65, 257)`), with unified tolerances (`rtol=2e-3, atol=2e-2` for fp16; `2e-2, 5e-2` for bf16; `1e-4, 1e-4` for fp32). The sibling `test_linear_batch_invariant_nd_precision` covers the convenience wrapper on 2D/3D/4D activations.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_invariant_ops.py -k linear_persistent
```
