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
- **Supported modes**: Input layout determines eligibility for head-scatter combine. The device profile supplies the head-scatter kernel's compile-time mode. The wrapper runs after DCP exchange in eager execution; the enclosing registered custom op supplies a FakeTensor implementation for tracing.

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

## Head-scatter implementation

BF16 head scatter uses the optimized kernel when `D <= 512`, the caller requests
only the output, and the optional local output has unit feature stride. Other
supported inputs use the scalar kernel.

Each program owns a contiguous interval. A3 processes one row at a time
because multirow LSE loads can lower to nested scalar loops. A5 loads each
LSE field across eight rows and all ranks, improving memory throughput.
The final tile masks rows outside the output. Other profiles use single-row mode.

Runtime and warmup use the same launch builder. The device profile supplies
`MODE` at compile time. `DCP_SIZE`, `BLOCK_D` and `HAS_LOCAL` specialize the
rank vector, feature width and optional local contribution. Sizes and strides
use Triton's ordinary scalar specialization.

## Test Cases

The single-card suite checks pack and combine against an FP32 reference.
Four head-scatter cases also compare independently packed inputs to a CPU FP64
reduction. They cover a single rank and feature, received-only inputs, both
local layouts, masked rank and feature lanes, and invalid or extreme LSE.
Outputs must be finite, at least 99% of elements must satisfy
`abs_error <= 2e-5 + 0.004 * abs(reference)`, and maximum absolute error must
be at most 0.02.

The warmup tests check that runtime and registration select the same mode.
The multi-card test exercises the registered operator with HCCL exchange.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -q tests/ut/model_executor/warmup/test_kernel_warmup.py
pytest -q tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py
```
