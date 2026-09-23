# GroupedMatmulSituQuant

`GroupedMatmul(MXFP8 x MXFP4) + SiTU + dynamic MX quant` single-launch fused
custom op for **Ascend950PR (arch35) only**, imported from PR #15871.
Fuses the split chain `npu_grouped_matmul + situ_mx_quant`.

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
- Only the MX A8W4 combo is implemented (Kimi w4a8); `bias`/`smoothScale`
  unsupported by design

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

Focused CPU tests are in `tests/ut/device/test_grouped_matmul_situ_quant.py` and
`tests/ut/quantization/methods/w4a8/test_w4a8_mxfp4.py`.
