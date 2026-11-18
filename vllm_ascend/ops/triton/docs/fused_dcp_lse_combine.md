# fused_dcp_lse_combine

> Source: `vllm_ascend/ops/triton/dcp/dcp_a2a.py` and `dcp_a2a_batched.py`.

## Description

- **Function**: Merges packed partial attention outputs received from DCP ranks and, when supplied, one local contribution. It reconstructs the packed LSE and performs a numerically stable weighted reduction in FP32 before casting to the receive-buffer dtype.
- **Formula**: For each output row, let valid contributions be `i` with finite `lse_i`, including the optional local contribution. With `m = max_i(lse_i)`, `w_i = exp(lse_i - m)`, the output is `sum_i(w_i * output_i) / sum_i(w_i)`. When `return_lse=True`, append `m + log(sum_i(w_i))`. If every contribution is invalid, output zero and, when requested, negative-infinite LSE.
- **Algorithm flow** (processed row by row, independently):
  1. Validate a contiguous receive buffer `[P, local_scatter_size, replicated_size, D + lse_pack_dim]`, infer the output's token/head shape from `scatter_dim`, and optionally validate the local tensors.
  2. Launch `min(output_rows, get_vectorcore_num())` Triton programs. The scalar kernel loops over rows, finds the largest valid LSE across the ranks and optional local contribution, then loops over ranks again to accumulate weighted output in FP32.
  3. Ignore non-finite LSE and mask its output before multiplication, preventing a zero weight times NaN from contaminating the result. Normalize once, store the output, and optionally append merged LSE.
  4. For eligible BF16 head-scatter inputs, the row-weighted kernel decodes each LSE once and reuses the rank weights across the feature dimension.
- **Supported modes**: All device profiles use the same input-based dispatch. The wrapper runs after DCP exchange in eager execution; the enclosing registered custom op supplies a FakeTensor implementation for tracing.

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
- `return_lse=True` requires an FP32 receive buffer. This is an inference operator without a backward pass.

## Origin and Differences

- **Origin**: Developed for the vllm-ascend DCP path to merge the payload produced by [`pack_dcp_output_lse`](pack_dcp_output_lse.md) after HCCL exchange.
- **Differences**:
    - NPU adaptation for performance: the row-weighted kernel streams rank contributions, limiting live FP32 accumulator storage. The scalar path retains the broader shape and dtype support.
    - Modified for vllm-ascend DCP flows: an optional local contribution is included exactly once after the received ranks; the wrapper can return a merged FP32 LSE for a later merge.

## Test Cases

The single-card nightly test compares the result against an independent FP32 softmax-weighted reference and the scalar kernel. It covers DCP sizes 1/2/8 and wider values for pack-only fallback, head dimensions 96/256/512, both scatter axes, strided local tensors, invalid LSE and dispatch boundaries. The combine comparison uses `assert_dcp_close`: at least 99% of elements meet the dtype's mixed tolerance, and the maximum absolute error also meets its limit. Defined nonfinite LSE sentinels are checked separately for exact equality. The multi-card test runs the registered operator with a real HCCL exchange.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py
```

## Row-weighted implementation

The wrapper selects the row-weighted kernel for BF16 head scatter with
`D <= 512`, no returned LSE, and unit local feature stride when a local
contribution is present. The same rule applies to every row count and device
profile. Other supported inputs use the scalar kernel.

Each program receives a balanced contiguous row interval. For one row, the kernel
decodes the received LSE fields once and places the optional local LSE in the same
small rank vector. It computes one maximum, exponent vector and denominator,
then streams each rank's output into one FP32 feature accumulator. The optional
local output uses its already-computed vector weight. Invalid contributions are
masked before multiplication; all-invalid rows produce zeros. The caller's
inputs are unchanged and the allocated output is fully written.

### Specialization and warmup

`DCP_SIZE`, `BLOCK_D` and `HAS_LOCAL` determine rank-vector shape, feature lanes
and the optional contribution, so they remain compile-time parameters. The
rank vector is `next_power_of_2(DCP_SIZE + HAS_LOCAL)`.

Sizes and strides remain runtime scalars, subject to ordinary Triton integer
specialization classes; pointer dtype/alignment and compiler options can also
create variants. Tensor contents and the grid alone do not create variants.
The existing DCP warmup uses the same production launcher and now includes
`_fused_dcp_lse_combine_bf16_head_scatter_kernel` in its wrapper inventory. Unit tests check
that runtime and warmup select the same combine kernel for small and large row counts. Serving
compilation counts remain unmeasured.

### Row-weighted validation

`test_dcp_combine_row_weights.py` compares public output with independently
encoded inputs and CPU FP64 weighted reduction. It includes trace-derived logical
shapes `(144,12,512,8)`, `(152,12,512,8)` and `(114,8,512,8)` as well as smaller
worlds, D384, non-power-of-two worlds, invalid/extreme LSE and strided local data.
The test requires at least 99% of BF16 elements to satisfy
`abs_error <= 2e-5 + 0.004 * abs(reference)`, with maximum absolute error at most
0.02 and every output finite. Invalid-LSE cases are handled explicitly. These are
reconstructed inputs, not model payload replay. A5, distributed HCCL and
model-level validation remain outstanding.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_combine_row_weights.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -q tests/ut/model_executor/warmup/test_kernel_warmup.py
```
