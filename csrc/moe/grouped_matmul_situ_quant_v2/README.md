# GroupedMatmulSituQuantV2 (A3 MSD backend)

This is the Ascend A3 backend for the existing `_C_ascend.grouped_matmul_situ_quant`
and `_C_ascend.grouped_matmul_situ_quant_weight_nz` Torch operators, including
their `.list` overloads. It fuses INT8/INT4 grouped matmul, SiTU-GLU and per-token
INT8 quantization. The arch32 implementation is ported from
[PR #59](https://github.com/hust17yixuan/vllm-ascend-kimi/pull/59), commit
`d956f8bc84b022bf042cc433cbbb7d7ee9e95a66`.

## Dispatch and compatibility

The A3 model route is opt-in through
`--additional-config '{"ascend_fusion_config":{"fusion_ops_gmmsituquant":true}}'`.
It defaults to the existing split path because the initial K3.1 operator
measurements do not establish a consistent performance benefit. The flag has
no effect on the existing A5 MX path. Direct Torch calls remain available
independently of the model flag.

- The public schemas and argument order are unchanged. INT8 activation selects
  this backend; FP8 activation retains the existing A5 implementation.
- Only `ascend910_93` builds this operator. The model route requires the
  `GROUPED_MATMUL_SITU_QUANT_MSD` capability, which is enabled only for A3.
- The internal GE prototype is `GroupedMatmulSituQuantV2`. The packed-weight
  bridge `aclnnGroupedMatmulSituQuantWeightNzV2` reinterprets INT32/INT64 carriers
  as INT4/UINT64 before invoking its generated ACLNN entry.
- A5 continues to call `aclnnGroupedMatmulSituQuant` with its original five
  inputs and three attributes. Its kernel, tiling and GE prototype are unchanged;
  a new A3 symbol is not required to import or use the A5 FP8 path.
- W4A8 model loading already produces native NZ weights. Dispatch uses the
  loader's stacked/EPLB contract, not the logical tensor rank to infer NZ format.
  Other hardware and unsupported model metadata retain the split path.

## A3 input contract

| Argument | Contract |
| --- | --- |
| `x` | Contiguous INT8 `[M_cap, K]`, `K > 0`, `K % 64 == 0` |
| `x_scale` | Contiguous FP32, one value per row |
| `weight` | Packed INT4 in INT32, logical `[E, K, N/8]`, or E tensors `[K, N/8]` |
| `weight_scale` | FP32 scale bits carried in INT64, `[E, N]`, or E tensors `[N]` |
| `weight_assist_matrix` | FP32 `8 * sum_k(weight) * weight_scale`, with the same expert form |
| `group_list` | Contiguous device INT64 `[E]`, counts (1) or cumulative counts (0) |
| Modes | `dequant_mode=0`, `dequant_dtype=0`, `quant_mode=0` |
| Activation | Finite `beta > 0`; finite `linear_beta`, with values <= 0 disabling up scaling |

All tensors must share `x`'s device. `N > 0` and `N % 128 == 0` are required.
ND and native NZ entries enforce their respective format contracts. `bias` and
`smooth_scale` are unsupported and rejected. Omitting the assist matrix selects
in-kernel compensation; the model uses its precomputed matrix.
Packed weight views must be contiguous with zero storage offset, matching the
loader's owned stacked tensors and cloned per-expert tensors.

The output is INT8 `[M_cap, N/2]` plus FP32 `[M_cap]`. Only the routed rows are
valid; padding is unspecified. Counts must be nonnegative and their sum must
not exceed `M_cap`. Routing values stay on device during capture and replay.
Zero-capacity inputs return empty outputs without a kernel launch.

MSD rounds two INT4 matrix products to FP16, merges them with compensation in
FP32, and applies SiTU before quantization. The existing split model path also
rounds the merged intermediate to BF16. Accordingly, accuracy validation checks
both an independent MSD reference and the production split chain; byte equality
with the BF16 intermediate is not the numerical contract.

## Validation

- `test_w4a8_situ_route.py`: A3 dispatch, A2/A5 fallback, EPLB and metadata guards.
- `test_grouped_matmul_situ_quant_meta.py`: both output contracts, all four public
  overloads, zero capacity and single/multiple experts; no device allocation.
- `test_gmm_situ_quant_msd.py`: ND/NZ, stacked/list, supplied/computed compensation,
  full INT8 inputs, sparse/empty routing, graph replay, invalid metadata and
  the production split reference including `E=14, K=N=7168, M_cap=7168`.
  Each reference receives its own input because the CANN A8W4 GMM rewrites its
  input while preparing MSD nibbles.
- `test_grouped_matmul_situ_quant.py`: existing A5 byte-exact split-reference
  regression with changing device routing under ACLGraph.
- `test_grouped_matmul_situ_quant_mx_compat.py`: native A5 checks for all four
  shared interfaces, including empty capacity and single/multiple experts.

Run `python benchmarks/benchmark_gmm_situ_quant_msd.py --output /tmp/situ-bench`
from a compiled A3 source checkout for the fixed K3.1 operator case matrix.
The benchmark validates eager/graph outputs, warms both paths, alternates their
measurement order and writes all samples and dispersion. It includes identical
input resets in both graphs and does not measure model serving performance.
