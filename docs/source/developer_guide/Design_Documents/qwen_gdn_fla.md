# Qwen GDN integration with flash-linear-attention-npu

The authoritative design is
[`2026-08-29-qwen35-qwen36-gdn-fla-design.md`](../../../superpowers/specs/2026-08-29-qwen35-qwen36-gdn-fla-design.md).
This page is a short developer-guide entry, not a second specification.

## Scope

Qwen3.5 and Qwen3.6 share one vLLM GDN model path. Eligible eager BF16
requests on A2, A3, and A5 can use operators from
`flash-linear-attention-npu` through `fla_npu.ops.ascendc`.

The FLA API and operator source are shared, but the installed wheel and custom
OPP must match the actual SoC:

| Hardware | FLA build target | vLLM-Ascend device family |
| --- | --- | --- |
| A2 | `ascend910b` | `AscendDeviceType.A2` |
| A3 | `ascend910_93` | `AscendDeviceType.A3` |
| A5 | `ascend950` | `AscendDeviceType.A5` |

310P and unknown device families retain the existing native GDN path.

## Execution topology

The preferred prefill core is the single fused entry
`fla_npu.ops.ascendc.gdn_core_fwd_phase6`. It replaces Python orchestration of
the following six prefill stages:

```text
chunk_local_cumsum
-> chunk_scaled_dot_kkt
-> solve_tri
-> recompute_w_u_fwd
-> chunk_gated_delta_rule_fwd_h
-> chunk_fwd_o
```

The six standalone entries remain a legacy and diagnostic composition. Causal
convolution and Q/K normalization remain outside the fused core. Ordinary
decode uses `recurrent_gated_delta_rule` when its FLA implementation is
selected. Output normalization, gating, and projection stay in the model layer.

## Eligibility and backend behavior

`get_fla_gdn_soc()` is the hardware capability boundary. The FLA adapter also
requires BF16 activations, ordinary non-speculative execution, PCP world size
one, and execution outside ACL Graph capture. Tensor parallelism is allowed.

The backend modes are:

- `auto`: resolve eligible FLA operators and otherwise retain the applicable
  native or legacy path;
- `fla_npu`: require the configured FLA symbols and fail with attribution when
  they cannot be selected;
- `native`: preserve the original vLLM-Ascend path.

One current limitation is important: after Phase 6 resolves successfully, a
failure in its first live-shape runtime probe does not yet restart the request
through the six-stage composition. The failure is logged and propagated. See
the authoritative design for the exact resolution, probe, and fallback
semantics.

Global strict `fla_npu` mode currently validates every Stage 1 replacement
symbol, including standalone prefill entries that the selected fused execution
graph may not call. A wheel containing the fused Phase 6 symbol alone is
therefore insufficient for strict startup.

## Validation

完整的 A2/A3 Docker 源码安装、FLA wheel 构建、单算子测试和
Qwen3.6 35B DP1/TP1 启动步骤，参见
[`Qwen3.6 35B GDN Phase6 A2/A3 部署与验证操作指南`](../../../superpowers/guides/2026-08-29-qwen-gdn-a2-a3-validation-guide-zh.md)。

Install the FLA wheel built for the target SoC, then run on each device family:

```bash
cd /home/z00886386/vllm-ascend

pytest -q tests/ut/device/test_device_config.py
pytest -q tests/ut/ops/test_gdn_fla.py
pytest -s -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py
```

Run the Qwen3.5 and Qwen3.6 eager model smokes separately. Selection logs must
show `gdn_core_fwd_phase6` and the expected `soc` value. Any fallback or failure
must identify the logical operator, concrete FLA symbol, stage, requested
backend, and SoC.

A2 has formal FLA Phase 6 evidence. A3 build support is not an A3 device
acceptance claim; A3 operator and model tests remain mandatory. A5 must run the
same regression matrix.

## Compatibility

The implementation lives in `vllm_ascend.ops.gdn_fla`. The former
`vllm_ascend.ops.gdn_a5` module and A5-prefixed class names remain temporary
import aliases. New code must use `FlaGDNAdapter` and
`FlaGDNOperatorDispatcher`.
