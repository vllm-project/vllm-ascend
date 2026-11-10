# fused_dcp_lse_combine

> Source: `vllm_ascend/ops/triton/dcp/dcp_a2a.py` and `dcp_a2a_batched.py`.

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
