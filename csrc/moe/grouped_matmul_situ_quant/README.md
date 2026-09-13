# grouped_matmul_situ_quant (GroupedMatmulSituQuant A5 fused op)

`GroupedMatmul(MXFP8 x MXFP4) + SiTU + dynamic MX quant` single-launch fused
custom op for **Ascend950PR (arch35) only**. Fuses the production split chain
`npu_grouped_matmul + situ_mx_quant` with bit-exact outputs; contract geomean
2.650x (8/8 >= 1.0), graph-mode ratio 1.06-1.96, delivered 2026-08-25.

## Layout

- `op_kernel/gmm_situ_vcv_dev.cpp` — device kernel (device group_list,
  graph-capturable static grid, in-kernel pruning)
- `op_kernel/gmsq_vcv_controller.h`, `op_kernel/situ_epilogue.h` — controller + SiTU/MXQuant epilogue
- `op_kernel/vendor/{wqbmm,gmsq2}` — vendored official arch35 weight-quant VCV
  data path (self-contained, no external deps)
- `op_host/gmm_situ_quant_entries.cpp` — host tiling/launch; the four
  V2-aligned entries (aclnnGroupedMatmulSwigluQuantV2 API habits; our op
  itself carries no version suffix)
- `ops.h` / `csrc/torch_binding.cpp` — `_C_ascend` dispatcher registration;
  `csrc/torch_binding_meta.cpp` supplies graph/compile Meta implementations
- Only the MX A8W4 combo is implemented (Kimi w4a8); `bias`/`smoothScale`
  unsupported by design

## Build / use

The kernel is built and packaged with `vllm_ascend_C` by the normal
vLLM-Ascend install on `SOC_VERSION=ascend950*`. Python surface:
`vllm_ascend.ops.grouped_matmul_situ_quant` (SOC gate, ND/NZ dispatch,
`to_weight_nz` helpers). Tests:
`tests/ut/ops/test_grouped_matmul_situ_quant.py`.

## Source of truth & sync

Kernel evolution happens in the external evolution delivery tree. Point
`GMSQ_P0_KERNEL` at its `p0_entries/kernel` directory before syncing an
accepted kernel change:

```bash
GMSQ_P0_KERNEL=/path/to/p0_entries/kernel \
    bash csrc/moe/grouped_matmul_situ_quant/sync_from_p0.sh
pip install -e .
pytest tests/ut/ops/test_grouped_matmul_situ_quant.py
```

`ops.h` and the root `CMakeLists.txt` integration are not synced.
