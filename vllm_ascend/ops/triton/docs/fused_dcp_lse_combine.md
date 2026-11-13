# fused_dcp_lse_combine

> Source: `vllm_ascend/ops/triton/dcp/dcp_a2a.py`, `dcp_a2a_batched.py`, and `dcp_a2a_rank_tiled.py`.

## Description

- **Function**: Merges packed partial attention outputs received from DCP ranks and, when supplied, one local contribution. It reconstructs the packed LSE and performs a numerically stable weighted reduction in FP32 before casting to the receive-buffer dtype.
- **Formula**: For each output row, let valid contributions be `i` with finite `lse_i`, including the optional local contribution. With `m = max_i(lse_i)`, `w_i = exp(lse_i - m)`, the output is `sum_i(w_i * output_i) / sum_i(w_i)`. When `return_lse=True`, append `m + log(sum_i(w_i))`. If every contribution is invalid, output zero and, when requested, negative-infinite LSE.
- **Algorithm flow** (processed row by row, independently):
  1. Validate a contiguous receive buffer `[P, local_scatter_size, replicated_size, D + lse_pack_dim]`, infer the output's token/head shape from `scatter_dim`, and optionally validate the local tensors.
  2. Launch `min(output_rows, get_vectorcore_num())` Triton programs. The scalar kernel loops over rows, finds the largest valid LSE across the ranks and optional local contribution, then loops over ranks again to accumulate weighted output in FP32.
  3. Ignore non-finite LSE and mask its output before multiplication, preventing a zero weight times NaN from contaminating the result. Normalize once, store the output, and optionally append merged LSE.
  4. On eligible BF16 head-scatter shapes `_fused_dcp_lse_combine_batched_kernel` processes eight rows per tile. It streams ranks to keep one `[BLOCK_ROWS, BLOCK_D]` FP32 accumulator live and masks the last partial tile.
- **Supported modes**: A3 and A5 share scalar and batched kernels selected by dtype, shape and strides. The wrapper runs after DCP exchange in eager execution; the enclosing registered custom op supplies a FakeTensor implementation for tracing.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `recv` | Input | Contiguous rank-major payload `[P, local_scatter_size, replicated_size, D + lse_pack_dim]` | BF16 / FP16 / FP32 | ND |
| `head_dim` | Input (attribute) | Number of output features, `D` | positive int | scalar |
| `scatter_dim` | Input (attribute) | `0` means token scatter; `1` means head scatter | int | scalar |
| `return_lse` | Input (attribute) | Append merged LSE to the last dimension | bool | scalar |
| `local_output` | Optional input | One additional partial output `[local_tokens, local_heads, D]`; may be strided | BF16 / FP16 / FP32 | ND |
| `local_lse` | Optional input | LSE for `local_output`, `[local_tokens, local_heads, 1]` | FP32 | ND |
| `output` | Output | Merged output `[local_tokens, local_heads, D + int(return_lse)]` | same as `recv` | ND |

## Constraints

- `recv` must be contiguous on NPU. Its packed dimension must equal `D + 1` for FP32 or `D + 4` for BF16/FP16. `local_output` and `local_lse` must be supplied together and match the post-scatter output shape.
- `return_lse=True` requires an FP32 receive buffer. The batched path requires `return_lse=False`; all other valid combinations use the scalar kernel.
- Batching requires BF16, `scatter_dim=1`, `D <= 512`, at least `4 * get_vectorcore_num()` output rows, and unit feature stride when `local_output` is present. Rank count has no hardware-specific upper bound.
- `BLOCK_D = next_power_of_2(D)` and `BLOCK_ROWS = 8`. Both local and received-only modes stream ranks. Unrolling is bounded by the live-element budget; multibuffering is disabled. This is an inference operator without a backward pass.

## Origin and Differences

- **Origin**: Developed for the vllm-ascend DCP path to merge the payload produced by [`pack_dcp_output_lse`](pack_dcp_output_lse.md) after HCCL exchange.
- **Differences**:
    - NPU adaptation for performance: the shared kernel batches rows while streaming rank contributions, limiting live FP32 accumulator storage. The scalar path retains the broader shape and dtype support.
    - Modified for vllm-ascend DCP flows: an optional local contribution is included exactly once after the received ranks; the wrapper can return a merged FP32 LSE for a later merge.

## Test Cases

The single-card nightly test compares the result against an independent FP32 softmax-weighted reference and the scalar kernel. It covers DCP sizes 1/2/8 and wider values for pack-only fallback, head dimensions 96/256/512, both scatter axes, strided local tensors, invalid LSE, shared dispatch boundaries, and a partial eight-row tile. The combine comparison uses `assert_dcp_close`: at least 99% of elements meet the dtype's mixed tolerance, and the maximum absolute error also meets its limit. Defined nonfinite LSE sentinels are checked separately for exact equality. The multi-card test runs the registered operator with a real HCCL exchange.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py
```

Eligible BF16 head-scatter shapes with a contiguous local contribution may use the rank-tiled kernel ahead of the batched path. Its parallel FP32 reduction can round differently from the scalar kernel. The rank tile uses one 16x512 BF16 tile per job, so the FP32 temporaries stay at 32 KiB. It requires a compile-time rank-row stride to vectorize; with runtime head/token counts A3 lowered it to scalar-rate transfers and ran far slower than the scalar kernel.

Device routing:

- **A5**: the dynamic kernel is used for DCP8/DCP16 with head dims 512 and 1024, where it measured 2.1-3.9x faster than the scalar and batched paths on the operator (including T64/T256). On a real 8-rank HCCL run the combine fell from 6.18/7.64/12.33/21.67us to 3.45/4.73/6.90/11.67us at T4/T16/T16-H6/T64, while the all-to-all stayed near 53us.
- **A3**: the dynamic kernel is never selected. Instead, warmup compiles a static twin (`NUM_HEADS`/`NUM_TOKENS` constexpr) for the engine's ACLGraph capture sizes (falling back to the #16350 decode budget `[1,2,4,8,16,32,48,64,96,128,192]`), and the dispatcher uses it only when the exact `(heads, tokens, head_dim, dcp, local strides, lse strides)` signature was warmed. Decode batches are padded to those graph sizes, so normal serving stays on the static kernel. Prefill token counts and any unwarmed signature fall back to the scalar/batched path with a one-time log instead of compiling inside a request. A3 DCP16 measured 2.0-2.2x faster than scalar/batched at D512 and 1.35x at D1024; DCP8 measured 1.40x at T16.

Warmup keys accumulate across target/draft warmup passes; a candidate that produced no launch is never installed, so a missed bucket can only fall back, not compile.
