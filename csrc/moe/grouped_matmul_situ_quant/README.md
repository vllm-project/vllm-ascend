# GroupedMatmulSituQuant

`GroupedMatmul + SiTU + dynamic quant` single-launch fused custom op, imported
from PR #15871 and extended with an Atlas A2/A3 path. Fuses the split chain
`npu_grouped_matmul + situ_mx_quant`.

Two data paths share one prototype, selected by the dtype of `x`:

- **Ascend950PR (arch35)** — MX A8W4: `MXFP8 x MXFP4`, e8m0 scales.
- **Atlas A2 / A3 (arch32)** — A8W4 MSD: `INT8 x INT4`, packed uint64 weight
  scale, per-token int8 + float32 output. The int32 packed int4 weight accepted
  by torch is rewritten into an INT4 descriptor by
  `op_host/op_api/aclnn_grouped_matmul_situ_quant_v2.cpp`.

## Layout

- `op_kernel/grouped_matmul_situ_quant.cpp` — A5 device kernel entry
  (`GMMSQWeightQuantResplitController` + `GmmVecActivationSitu`; device
  group_list, graph-capturable static grid, in-kernel pruning)
- `op_kernel/weight_quant_basic_block/` — A5 (arch35) MX A8W4 kernel body,
  byte-synced from ops-transformer's `gmm/grouped_matmul_situ_quant`, so the
  data path matches the released op exactly; self-contained apart from the
  CANN public headers
- `op_kernel/{situ_activation.h,grouped_matmul_situ_quant_tiling_key.h,grouped_matmul_situ_quant_arch35_tiling_data.h}`
  — SiTU activation VF, `ASCENDC_TPL` tiling-key declaration and the
  arch35 tiling-struct alias
- `op_kernel/grouped_matmul_situ_quant_apt.cpp` + `op_kernel/arch32/` — A2/A3
  A8W4 MSD kernel (arch32), registered through its own `opFile.value`
  (the `_apt` suffix follows the repo's second-entry convention, e.g.
  `grouped_matmul_swiglu_quant_v2_apt.cpp`, and is auto-resolved by
  `add_bin_compile_target`)
- `op_host/grouped_matmul_situ_quant_tiling.cpp` — host tiling entry: dispatches
  on `x` dtype (INT8 → A2/A3 A8W4 template, otherwise → arch35)
- `op_host/grouped_matmul_situ_quant_arch35_tiling.{h,cpp}` — A5 (arch35)
  tiling, collected by the standard `op_host/*_tiling*.cpp` glob
- `op_host/grouped_matmul_situ_quant_a8w4_*.{h,cpp}` — A2/A3 A8W4 MSD tiling
  implementation, collected by the standard `op_host/*_tiling*.cpp` glob
- `op_host/op_api/` — `aclnnGroupedMatmulSituQuantV2`: A2/A3 int32 packed int4
  weight entry, delegating to the auto-generated `aclnnGroupedMatmulSituQuant`
- `grouped_matmul_situ_quant_torch_adpt.h` / `csrc/torch_binding.cpp` — `_C_ascend` dispatcher registration;
  `csrc/torch_binding_meta.cpp` supplies graph/compile Meta implementations
- Only the A8W4 combo is implemented (Kimi w4a8); `bias`/`smoothScale`
  unsupported by design

## Build / use

The kernel is built and packaged with `vllm_ascend_C` by the normal
vLLM-Ascend install on `SOC_VERSION=ascend950*` (A5 / arch35) or on
`ascend910b*` / `ascend910_93*` (A2/A3 / arch32). The existing
`DeviceOperator.npu_grouped_matmul_situ_quant` entry calls
`torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz` on A5.
Tensor-list dispatch uses `.list`; ND entries retain the reference names.

The W4A8 MXFP MoE method automatically selects fusion for SiTU, group size 32,
group-list types 0 (cumulative) or 1 (counts), and positive `linear_beta`.
W13 is loaded with native FP4 metadata; scales use E8M0 metadata and N-major
views without copying storage. W2 and the existing GMM2 path are unchanged.

The Ascend 950 end-to-end case is in
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_grouped_matmul_situ_quant.py`.
