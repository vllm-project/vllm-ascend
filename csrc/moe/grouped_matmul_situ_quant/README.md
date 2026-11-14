# GroupedMatmulSituQuant

Single-launch grouped matmul, SiTU activation and output quantization for
Ascend A3 and A5. A3 uses INT8/INT4 MSD with per-token INT8 output; A5 uses
MXFP8/MXFP4 with dynamic MX output quantization.

## Layout

- `op_kernel/grouped_matmul_situ_quant.cpp` — device kernel (device group_list,
  graph-capturable static grid, in-kernel pruning)
- `op_kernel/gmsq_vcv_controller.h`, `op_kernel/situ_epilogue.h` — controller + SiTU/MXQuant epilogue
- `op_kernel/vendor/{wqbmm,gmsq2}` — vendored official arch35 weight-quant VCV
  data path (self-contained, no external deps)
- `op_host/grouped_matmul_situ_quant_tiling.cpp` — host tiling; the four
  V2-aligned entries (aclnnGroupedMatmulSwigluQuantV2 API habits; our op
  itself carries no version suffix)
- `grouped_matmul_situ_quant_torch_adpt.h` / `csrc/torch_binding.cpp` — `_C_ascend` dispatcher registration;
  `csrc/torch_binding_meta.cpp` supplies graph/compile Meta implementations
- `op_kernel/grouped_matmul_situ_quant_apt.cpp` and `op_kernel/arch32/` —
  A3 MSD implementation, selected by the A3 operator configuration
- `op_host/grouped_matmul_situ_quant_a8w4_base_tiling.cpp` — A3 tiling
- `bias` and `smoothScale` are unsupported

## Build / use

The kernel is built and packaged with `vllm_ascend_C` by the normal
vLLM-Ascend install on `SOC_VERSION=ascend950*`. The existing
`DeviceOperator.npu_grouped_matmul_situ_quant` entry calls
`torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz` on A5.
Tensor-list dispatch uses `.list`; ND entries retain the reference names.

The W4A8 MXFP MoE method automatically selects fusion for SiTU, group size 32,
group-list types 0 (cumulative) or 1 (counts), and positive `linear_beta`.
W13 is loaded with native FP4 metadata; scales use E8M0 metadata and N-major
views without copying storage. W2 and the existing GMM2 path are unchanged.

The Ascend 950 end-to-end case is in
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_grouped_matmul_situ_quant.py`.

## Ascend A3

The INT W4A8 MoE method follows the same `DeviceOperator` integration as A5.
For SiTU with per-channel weights, a weight compensation matrix, group-list
types 0 or 1, K divisible by 64 and N divisible by 128, A3 selects the fused
operator. Other cases retain the split GMM and activation path. No additional
configuration flag is required.

The shared Torch interfaces accept either a stacked expert tensor or `.list`
expert tensors. A3 uses INT8 `[M, K]` activations, packed INT32 weights (eight
INT4 values per carrier), INT64 weight-scale bit carriers and FP32 per-token
scales. `weight_assist_matrix` carries the existing FP32 compensation
`8 * sum(weight * weight_scale)`; `dequant_mode=0` and `quant_mode=0` select
per-channel dequantization and per-token output quantization. Output is INT8
`[M, N/2]` plus FP32 `[M]`. Routing remains on device during graph replay.

A3 and A5 share the GE operator registration. The A3 inputs and attributes
extend its native ACLNN signature; rebuild the custom OPP and framework
extension together on both platforms. Mixing an older OPP with the new
extension is unsupported. The A5 device computation and MX mode are retained.

Build the A3 operator with:

```bash
cd csrc
bash build.sh --pkg --soc=ascend910_93 --ops=grouped_matmul_situ_quant -j16 -O3
```

The A3 numerical and graph tests are in
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_situ_quant_msd.py`.

See [A3 profiling and validation](A3_PERFORMANCE.md) for implementation changes,
workspace sizing, measured gains and remaining split-path performance gaps.
