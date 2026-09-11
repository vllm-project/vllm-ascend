# l2norm_fwd_kernel2_loop

Source: `vllm_ascend/ops/triton/fla/l2norm.py` (host wrapper: `l2norm_fwd`).

## Description

- **Function**: Row-wise L2 normalization of the last dimension of a tensor. It is the forward-only NPU Triton kernel behind `l2norm_fwd`, used by the FLA (flash-linear-attention) chunked gated delta rule path to normalize Q/K before the attention computation (`use_qk_l2norm_in_kernel=True` in `vllm_ascend/ops/triton/fla/chunk.py`).
- **Formula**: For every row `x` of the flattened 2-D input (row length `N`, accumulation in fp32):

    ```text
    y[i, :] = x[i, :] * rsqrt(sum(x[i, :] ^ 2) + eps)
    ```

    which is equivalent to `torch.nn.functional.normalize(x, dim=-1, p=2)` with an `eps` added inside the square root.
- **Algorithm flow** (processed row by row, independently):
  1. Host side (`l2norm_fwd`): reshape the input to 2-D `[T, D]` (`T = prod(shape[:-1])`, `D = shape[-1]`), allocate the output (`output_dtype` if given, otherwise the input dtype) and check that `D` fits into a 64KB feature block (`BD = min(65536 // element_size, next_power_of_2(D))`, raise if `D > BD`).
  2. Host side: build a persistent grid of `num_core = get_vectorcore_num()` programs; each program owns `NUM_CHUNKS = cdiv(cdiv(T, num_core), MBLOCK)` chunks of `MBLOCK = 69` rows, i.e. it starts at `base_row = pid * NUM_CHUNKS * MBLOCK`.
  3. Kernel side: loop over the `NUM_CHUNKS` chunks. For each chunk, compute `row_idx = base_row + chunk * MBLOCK + arange(0, MBLOCK)` and the row mask `row_idx < M`; out-of-range rows are masked on both load and store, so an arbitrary `T` is supported.
  4. Kernel side: load the `[MBLOCK, N]` tile (masked, `other=0.0`), cast to fp32, compute `square_sum = sum(x * x, axis=1)`, then `rsqrt(square_sum + eps)`, multiply the tile by the per-row reciprocal norm and store it back to `Y` with the same mask.
  5. Host side: reshape the output back to the original input shape.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `X` | Input | Flattened 2-D input tensor `[M, N]` (view of the original `[..., D]` tensor), last dimension contiguous | fp32 / fp16 / bf16 | ND |
| `Y` | Output | Normalized output `[M, N]`, same shape as `X`; dtype is `output_dtype` if provided, otherwise `X.dtype` | fp32 / fp16 / bf16 | ND |
| `eps` | Input (attribute) | Epsilon added to the sum of squares before `rsqrt`, default `1e-6`; not specialized (`do_not_specialize`) | fp32 | scalar |
| `M` | Input (attribute) | Total number of rows `T = prod(shape[:-1])`; not specialized (`do_not_specialize`) | int32 | scalar |
| `N` | Input (attribute) | Feature dimension `D = shape[-1]`, compile-time `constexpr` | int32 | scalar |
| `MBLOCK` | Input (attribute) | Number of rows per chunk, compile-time `constexpr`, fixed to `69` by the wrapper | int32 | scalar |
| `NUM_CHUNKS` | Input (attribute) | Number of chunks processed by one program, `cdiv(cdiv(T, num_vectorcore), MBLOCK)`; not specialized (`do_not_specialize`) | int32 | scalar |

## Constraints

- `X` must have a contiguous last dimension; the wrapper enforces `y.stride(-1) == 1` on the output.
- `N` (`D`) must satisfy `D <= 65536 // X.element_size()`; a larger feature dimension raises `RuntimeError: l2norm_fwd: This layer doesn't support feature dim >= 64KB`.
- The whole `[MBLOCK, N]` tile must fit into UB; `MBLOCK` is fixed to `69` and `N` is bounded by the 64KB feature-block check above.
- `eps > 0` is expected (default `1e-6`); it only guards the `rsqrt` against all-zero rows.
- `M` is dynamic: row masks handle any `T`, including `T` not divisible by `MBLOCK * NUM_CHUNKS`, so dynamic token counts are supported.
- The reduction is always performed in fp32 regardless of the input dtype; the result is cast back on store, so fp16/bf16 inputs keep the accuracy of an fp32 accumulation.
- Forward only (inference); no backward pass is provided. Graph mode: supported — the grid depends only on the device vector-core count and host-side shapes, and the kernel itself contains no host synchronization.

## Origin and Differences

- **Origin**: Adapted from `vllm/model_executor/layers/fla/ops/l2norm.py` of vLLM (itself copied from the flash-linear-attention project, MIT license; see the file header).
- **Differences**:
    - NPU adaptation for performance: replaces the one-program-per-row-block grid with a persistent grid of exactly `get_vectorcore_num()` programs, where each program loops over `NUM_CHUNKS` chunks of `MBLOCK = 69` rows. This keeps every vector core busy and amortizes the kernel-launch and tail overhead of the original tiled version on Ascend NPUs;
    - Modified for a specific vllm-ascend logic or different input parameters: forward-only kernel (the backward pass and the `residual`/`bias` variants of the upstream file are dropped), `eps`/`M`/`NUM_CHUNKS` are marked `do_not_specialize` to avoid recompilation for every token count, and the wrapper keeps the upstream `l2norm_fwd(x, eps, output_dtype)` signature so it can be dropped into `vllm_ascend/ops/triton/fla/chunk.py` unchanged.

## Test Cases

> [!NOTE]
> Single-operator accuracy test cases are placed under `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`.

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_l2norm.py` compares the kernel against `torch.nn.functional.normalize(x, dim=-1, p=2)` over `(B, T, H, D)` shapes `(1, 63, 1, 60)`, `(2, 500, 4, 64)`, `(2, 1000, 2, 100)` and `(3, 1024, 4, 128)`, covering a row count that is not a multiple of `MBLOCK` as well as a non-power-of-two feature dimension. All current cases use fp32 with `rtol=3e-4, atol=1e-3`; fp16 and bf16 are listed as supported input dtypes above but are not covered by this test file yet.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_l2norm.py
```
