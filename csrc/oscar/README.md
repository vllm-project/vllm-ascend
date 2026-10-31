# OSCAR INT2 launcher extension sources

Two C++ torch-extension sources backing the INT2 three-tier KV path. They
are compiled at engine load time with `torch.utils.cpp_extension.load`
(`vllm_ascend/oscar/paths.py`: `launcher_src()` / `lse_bridge_src()`), or
ahead of time by any standard extension build for the same sources.

## Sources

| file | exported entry points (pybind) | role |
|------|-------------------------------|------|
| `int2_op_launch_lse_fast_r152.cpp` | `launch_lse_fast`, `verify_lse`, `warm_ws`, `oscar_middle_prep{,_batched,_batched_v2,_v3_fill}`, `oscar_middle_replay{,_batched}`, `oscar_middle_op_v3`, `oscar_quant_kv`, `oscar_scatter_int2`, `oscar_fill_v3`, `oscar_merge_tail`, `r144_read_counter` | production: INT2 middle-tier aclnn launches, single-call C++ K/V quantize (ATen-only; value-identical to the Torch encoder) and paged scatter |
| `int2_op_launch_lse_r98.cpp` | `launch_lse`, ... | R98 LSE bridge: standalone/diagnostic launcher for the sibling LSE op |

## Build & runtime dependencies (exact contract)

- torch / torch_npu headers (`torch_npu/include`, `.../third_party/op-plugin`)
  and `libtorch_npu` at link time; `c10_npu` Device/StreamGuards used on
  every raw pybind entry (see guards in `oscar_quant_kv`).
- CANN 9.0.0: `acl/acl.h`, `aclnn` headers, `libascendcl`, `libascend_hal`
  (driver) preloaded RTLD_GLOBAL before the first aclnn call.
- **Vendor operator package** (built from the `oscar-int2-ops` repository):
  `op_api/include/aclnn_oscar_int2_fused_attn_int2_lse.h` and
  `op_api/lib/libcust_opapi.so` providing
  `aclnnOscarInt2FusedAttnInt2Lse{GetWorkspaceSize,}` and the `...Tensor`
  variant. The launchers treat tiling buffers as opaque; the ABI the
  runtime must agree on is the vendor dispatch json `opParaSize` (2608 for
  the frozen ABI; see the ops repo `scripts/fix_op_para_size.py` for the
  provenance chain and the contract battery gating it).
- The OPP chain is resolved fail-closed from `R144_OPP_PATH` /
  `OSCAR_OPP_PATH` / `ASCEND_CUSTOM_OPP_PATH` (first entry = kernel
  vendor; see `oscar/paths.py`).

## Numerical-contract notes

- `oscar_quant_kv` is ATen-only (kthvalue/amin/amax/clamp/round/pack);
  bit-exact against `oscar/int2_quant.py` (`at::kthvalue(k) ==
  sorted[k-1]`, asserted in the encoder unit tests).
- Stream/device guards + `aclrtSynchronizeStream/Device` barriers are
  load-bearing (R113 stale-read garble class): keep them when refactoring.

## Verified build & load (2026-10-23, Ascend 910B, CANN 9.0.0, torch 2.7.0a0 / torch_npu)

Both sources were compiled from this tree through
`torch.utils.cpp_extension.load` in a fresh build directory, imported, and
exercised against the frozen vendor kernel
(`oscar_r144_v133c_grp`, kernel `.o` md5 `c8a8d47a63be7ad7`):

- `int2_op_launch_lse_fast_r152.cpp` — compiles clean; module imports;
  `r112fast_version() == "r113.orch.v9+r145v3"`.
- `int2_op_launch_lse_r98.cpp` — compiles clean; module imports;
  `r98_lse_version() == "r98.lse.v1"`.
- `oscar_quant_kv(k, v, 0.96, 0.92)` run on-device (64 tokens, 8 KV heads,
  head_dim 128): **bit-equal on all six outputs** (K/V × packed/scale/zero)
  to the pure-Torch encoder from PR 2 (`oscar_quantize_int2`) — the
  bit-parity contract between the C++ launcher and the Torch reference,
  demonstrated live in one process.

Inputs/outputs are SHA256-pinned in the evidence record
(`run_build_evidence.py` + `evidence.json` in the delivery workspace);
a compact reproduction recipe is `paths.py:launcher_src()` + the include/link
flags above.
