# Qwen3.5/Qwen3.6 cross-SoC FLA GDN integration design

## Status

- Date: 2026-08-29
- Repository: `vllm-ascend`
- Hardware scope: Ascend A2, A3, and A5
- Model scope: Qwen3.5 and Qwen3.6
- Primary prefill implementation: `fla_npu.ops.ascendc.gdn_core_fwd_phase6`
- Dependency: `flash-linear-attention-npu`, installed as `fla_npu`
- Current delivery level: eager BF16 ordinary prefill and ordinary decode

This is the authoritative design for the integration. It supersedes the
2026-08-25 A5-only, six-operator Stage 1 design.

## Decision summary

Qwen3.5 and Qwen3.6 use one shared GDN integration in vLLM-Ascend. On A2, A3,
and A5, eligible eager BF16 requests can use FLA operators. The normal prefill
core path is one Phase 6 fused call rather than six separately orchestrated
forward calls.

The fused call covers these six mathematical stages:

```text
chunk_local_cumsum
-> chunk_scaled_dot_kkt
-> solve_tri
-> recompute_w_u_fwd
-> chunk_gated_delta_rule_fwd_h
-> chunk_fwd_o
```

The phrase "fused GDN core" does not mean the entire Qwen GDN layer is one
operator. Causal convolution and Q/K normalization happen before the fused
prefill core. Ordinary decode uses a separate recurrent operator. Output norm,
gating, and projection remain in the model layer.

The six standalone prefill stages remain available as a legacy and diagnostic
composition. They are not the preferred prefill path when Phase 6 is selected.

## Goals

- Use one shared FLA GDN contract on A2, A3, and A5.
- Use `gdn_core_fwd_phase6` as the normal eager prefill core.
- Preserve the existing vLLM-Ascend path and the six-stage composition for
  compatibility, diagnosis, and fallback work.
- Use the public FLA recurrent GDN entry for ordinary decode when available.
- Keep backend selection and failures attributable to an exact symbol, SoC,
  model layer, and execution phase.
- Keep Qwen3.5 and Qwen3.6 on one implementation while requiring independent
  model-level acceptance.
- Reach MTP and ACL Graph through separately validated later stages.

## Non-goals of the current stage

- Training or backward propagation.
- MTP or other speculative decode through the FLA adapter.
- ACL Graph capture or replay through the FLA adapter.
- PCP execution with world size greater than one.
- Prefix caching acceptance.
- Changing upstream vLLM model definitions.
- Building or repackaging FLA inside vLLM-Ascend.
- Claiming that A3 is accepted only because its target can be compiled.

## Hardware and package contract

FLA uses the same Python API, ACLNN entry, and operator source for all three
hardware families. The binary package is SoC-specific.

| Hardware | FLA build target | vLLM-Ascend device family |
| --- | --- | --- |
| A2 | `ascend910b` | `AscendDeviceType.A2` |
| A3 | `ascend910_93` | `AscendDeviceType.A3` |
| A5 | `ascend950` | `AscendDeviceType.A5` |

Each environment must install a wheel and custom OPP produced for its actual
SoC. Sharing an A2 wheel with A3 or A5 is not supported. The wheel, Python
wrapper, op-api library, kernel package, and OPP metadata must come from one
compatible build.

`get_fla_gdn_soc()` is the vLLM-Ascend capability boundary:

- A2 returns `ascend910b`;
- A3 returns `ascend910_93`;
- A5 returns `ascend950`;
- 310P and unsupported devices return no FLA GDN target.

Hardware eligibility does not prove that the installed FLA package is usable.
Symbol resolution and live-shape probes provide the second capability layer.

## Shared model integration

Qwen3.6 reuses the upstream Qwen3.5 model implementation and the shared
`QwenGatedDeltaNetAttention` patch. No separate Qwen3.6 GDN Python layer is
introduced.

`vllm_ascend/ops/gdn.py` remains responsible for:

- projection output preparation;
- identifying prefill, ordinary decode, speculative decode, and mixed batches;
- owning convolution and SSM cache references;
- obtaining GDN metadata and cache indices;
- invoking the FLA adapter only for an eligible execution;
- scattering final states and restoring token order;
- completing output gating and projection.

`vllm_ascend/ops/gdn_fla.py` is responsible for:

- parsing global and per-operator backend modes;
- resolving public FLA symbols lazily;
- normalizing dtype, layout, metadata, and return contracts;
- selecting and caching implementations by runtime signature;
- running eager prefill, ordinary decode, causal convolution, and probes;
- emitting selection, fallback, probe, and execution-error logs.

`vllm_ascend/ops/gdn_a5.py` is a compatibility import module. New code uses the
FLA names and module. A5-prefixed class aliases remain only to avoid breaking
existing downstream imports immediately.

## Eligibility gate

The FLA adapter is created only when all current Stage 1 conditions hold:

- `get_fla_gdn_soc()` returns A2, A3, or A5;
- the activation dtype is BF16;
- `num_spec == 0`;
- PCP world size is one;
- the current forward is not inside graph capture;
- the effective backend configuration is not fully `native`.

If a strict FLA selection is requested with a non-BF16 activation, model setup
raises an unsupported-dtype error. In non-strict mode, the existing path is
used instead.

TP is not excluded by this gate. The runtime signature uses the per-TP-rank key
and value head counts. TP acceptance is still an explicit test obligation.

## Operator topology

| Layer segment | Preferred implementation | Existing/diagnostic path | Fused core member |
| --- | --- | --- | --- |
| Causal convolution | `fla_npu.ops.ascendc.causal_conv1d` | `npu_causal_conv1d_custom` | No |
| Q/K normalization | vLLM-Ascend native `l2norm_fwd` | Same implementation | No |
| Prefill core | `fla_npu.ops.ascendc.gdn_core_fwd_phase6` | Six-stage composition | Yes |
| Ordinary decode | `fla_npu.ops.ascendc.recurrent_gated_delta_rule` | vLLM-Ascend recurrent op | No |
| Output norm/gate/projection | Existing model code | Existing model code | No |

`l2norm_fwd` is the only intentionally native-only logical entry in the current
registry. A per-operator request for `l2norm_fwd=fla_npu` is rejected.

Backward-only operators are not imported or executed by serving:

- `prepare_wy_repr_bwd_full`;
- `chunk_gated_delta_rule_bwd_dhu`;
- `chunk_bwd_dv_local`;
- `prepare_wy_repr_bwd_da`;
- `chunk_bwd_dqkwg`.

## Prefill data flow

### Inputs owned by vLLM-Ascend

Before the FLA adapter, the logical tensors are:

```text
q, k:          [B, T, Hk, K]
v:             [B, T, Hv, V]
g, beta:       [B, T, Hv]
initial_state: [N, Hv, V, K]
```

The adapter validates that `Hv` is an integer multiple of `Hk`. Q and K are
normalized exactly once before selecting the fused or standalone prefill path.

### Phase 6 primary path

When `GDN_CORE_FWD` resolves to FLA, the adapter performs:

```text
native l2norm(q, k)
-> transpose q/k/v from BTHD to BHTD
-> clone and clear missing initial states
-> transpose state tail from [V, K] to [K, V]
-> gdn_core_fwd_phase6(...)
-> transpose output from BHTD to BTHD
-> transpose final state tail back to [V, K]
```

Phase 6 receives native GVA heads. Q and K are not repeated to `Hv` before the
call. The kernel handles GVA internally. The wrapper passes the original gate
values because local cumsum is inside the fused core. It converts `beta` to
FP32 to match the current OpDef contract.

The public FLA call returns:

```text
output, final_state, g_cumsum, A
```

Serving consumes `output` and `final_state`; the current wrapper discards
`g_cumsum` and `A`.

The current FLA Phase 6 wrapper requires:

- `K == 128`;
- `V` equal to 128 or 256;
- `chunk_size` equal to 64 or 128;
- FP16 or BF16 Q/K/V at the public wrapper boundary;
- `Hv % Hk == 0`;
- canonical sequence-major metadata for variable-length input.

The current vLLM-Ascend Stage 1 route additionally requires BF16 activations.

For variable-length input, `cu_seqlens_host` and `chunk_indices_host` are
converted to host integer lists and passed to FLA. They must describe the same
packed tokens as the device metadata. Dense execution omits both together.

### Six-stage legacy path

If no FLA core entry is added to the selected operator map, the adapter uses:

```text
chunk_local_cumsum
-> chunk_scaled_dot_kkt
-> solve_tri
-> recompute_w_u_fwd
-> chunk_gated_delta_rule_fwd_h
-> chunk_fwd_o
```

This path expands Q/K heads when `Hv > Hk`, uses BHTD intermediate layouts,
supports the existing `keep_meta` state filtering and reinsertion behavior, and
normalizes each standalone FLA/native contract separately.

The standalone path remains important for:

- comparing Phase 6 with the previous composition;
- isolating one mathematical stage;
- retaining existing metadata and state behavior;
- future whole-pipeline fallback work.

It is not evidence that the preferred Phase 6 request executed six separate
operators. Profiler or operator logs must identify `ChunkGdnCoreFwd` when a true
single-kernel Phase 6 run is required.

## Ordinary decode data flow

Ordinary non-MTP decode keeps Q/K normalization outside the recurrent operator:

```text
native l2norm(q, k)
-> fla_npu.ops.ascendc.recurrent_gated_delta_rule
-> update selected SSM cache blocks
```

The normalized recurrent wrapper accepts token tensors, gate, beta, scale,
`actual_seq_lengths`, and `ssm_state_indices`. It supports either an in-place
result or a functional `(output, final_state)` result; the latter is copied back
to the selected state blocks.

The native recurrent vLLM-Ascend operator remains the fallback implementation.
MTP accepted-token handling is not enabled through this adapter because the
adapter is bypassed when `num_spec > 0`.

Mixed ordinary decode and prefill keeps decode-first ordering: split ordinary
decode tokens, run recurrent decode, run prefill, and concatenate the outputs
back in the expected order.

## Backend configuration

### Global mode

`VLLM_ASCEND_GDN_BACKEND` accepts:

- `auto`: resolve FLA replacements and use native selections when resolution or
  a supported probe can fall back safely;
- `fla_npu`: strict mode; required resolution or probe failures are raised;
- `native`: bypass the FLA adapter when every effective selection is native.

### Per-operator mode

`VLLM_ASCEND_GDN_OP_BACKENDS` accepts comma-separated `operator=backend`
entries. `gdn_core_fwd` is a logical operator and can be overridden.

Standalone-stage overrides affect only the six-stage path. They cannot replace
an internal stage of an already selected Phase 6 kernel. To diagnose standalone
stages while the adapter remains active, `gdn_core_fwd` must first be selected
as native so the fused entry is not added to the pipeline.

Invalid names, duplicates, `auto` as a per-operator value, and a FLA override
for native-only l2norm fail configuration parsing.

## Selection, probing, and caching

Selections are cached by logical operator and a runtime signature containing:

```text
SoC, activation dtype, state dtype, Hk, Hv, K, V,
chunk size, MTP flag, ACL Graph flag
```

The dispatcher has three validation levels:

1. Resolve the public Python symbol.
2. Execute the first live-shape call as a runtime probe and synchronize the NPU.
3. Validate output structure, shape, dtype, finiteness, and state effects where
   applicable.

Stateful causal convolution probes clone their state before execution. After a
successful stateful probe, the live operation is executed once against the real
cache. Non-stateful probes return the already computed result without a second
execution.

Warmup constructs scratch prefill, causal-convolution prefill, and
causal-convolution decode calls. Warmup is cached across layers with the same
signature and convolution width.

## Fallback semantics and current limitations

Fallback behavior must be described at two different levels.

### Resolution-time behavior

If `gdn_core_fwd_phase6` cannot be resolved in `auto` mode, the FLA core entry
is not added to the operator map. Prefill then enters the six-stage path. Each
standalone stage has its own normalized FLA/native selection.

If strict mode cannot resolve a required symbol, model initialization fails
with the logical operator and first-line reason.

### Runtime-probe behavior

Stateful operators are never retried blindly after a live-cache execution
failure. Logs identify whether persistent state may have been mutated.

The current Phase 6 runtime-probe fallback is incomplete. Its native callback
deliberately raises because a safe fallback must restart through the standalone
pipeline. If the first Phase 6 live-shape probe fails after successful symbol
resolution, the current implementation raises instead of restarting the six
stages from the beginning. Documentation and test results must not describe
this case as a successful automatic whole-pipeline fallback.

Warmup currently uses the same probe phase keys as live traffic. A successful
scratch warmup can therefore mark a signature as already probed before the
first real request. A later shape-specific live failure then propagates without
running a new fallback probe. This is a known implementation gap and must not
be described as first-live-shape coverage.

A future code change may implement safe full-pipeline retry because Phase 6
receives a cloned initial state. That behavior is not part of the current
implementation and is not authorized by this documentation update.

### Strict symbol scope

Global strict mode currently validates every Stage 1 replacement symbol,
including the six standalone prefill entries, causal convolution, recurrent
decode, and fused core. Therefore a package containing only the Phase 6 symbol
is not sufficient for current global strict startup, even when the preferred
prefill execution would use only Phase 6.

Reducing strict validation to the selected execution graph is a future code
improvement, not current behavior.

## Logging and diagnosis

Selection logs use the `GDN FLA` prefix and contain:

- logical operator;
- selected backend and exact symbol;
- SoC and activation/state dtypes;
- Hk, Hv, K, V, and chunk size;
- MTP and ACL Graph flags.

Example:

```text
GDN FLA operator selected: op=gdn_core_fwd backend=fla_npu
symbol=fla_npu.ops.ascendc.gdn_core_fwd_phase6 soc=ascend910b
```

Fallback warnings include the requested backend, native selection, resolution
or probe stage, exception type, and concise reason. Execution errors additionally
include the model layer, phase, tensor metadata, and whether state may have been
mutated. Tensor values are not logged.

Seeing standalone stage selection logs during setup does not prove those stages
executed for prefill. Confirm the `gdn_core_fwd` selection and profiler kernel
name when validating the fused path.

## Validation strategy

### FLA dependency validation

For each SoC-specific wheel:

- confirm `gdn_core_fwd_phase6`, recurrent GDN, and required companion symbols
  import from the installed package;
- run the FLA official Phase 6 and recurrent tests;
- cover dense and variable-length metadata;
- cover native GVA and relevant head ratios;
- cover chunk sizes 64 and 128 where supported;
- compare output, final state, `g_cumsum`, and `A` with the reference path;
- record the loaded op-api and OPP package identity.

### vLLM-Ascend unit and operator tests

The repository tests cover:

- A2/A3/A5 capability mapping and unsupported-device bypass;
- backend parsing, strict failures, selection caching, and logs;
- Phase 6 input/output normalization;
- standalone-stage diagnostics;
- variable-length metadata and state layout;
- causal convolution cache update;
- ordinary recurrent decode and state update;
- native and FLA output/state comparison.

Current test paths are:

```text
tests/ut/device/test_device_config.py
tests/ut/ops/test_gdn_fla.py
tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py
```

### Model acceptance

Qwen3.5 and Qwen3.6 require independent real-weight eager smoke tests. Sharing
one model implementation does not allow one model result to stand in for both.

For each accepted hardware family:

- run native and FLA backends with deterministic greedy decoding;
- compare generated tokens and require non-empty output;
- cover TP1 where possible and TP2 where required by model size;
- confirm the expected physical devices and SoC-specific FLA package;
- inspect selection, fallback, and runtime-error logs;
- profile the prefill to prove the fused kernel executed.

Initial numerical criteria remain:

- output cosine similarity at least 0.999 for operator comparisons;
- BF16 `rtol=5e-3`, `atol=5e-3` unless recorded evidence justifies a change;
- final state compared separately;
- all outputs and states finite;
- deterministic smoke tokens equal to native.

## Current evidence boundary

| Hardware | Build support | Phase 6 evidence | vLLM model status | AISBench |
| --- | --- | --- | --- | --- |
| A2 | Yes | Formal FLA Phase 6 archive and focused GVA/dense-tail evidence | End-to-end validation pending | Not validated |
| A3 | Yes | No Phase 6 device evidence identified in the inspected branch | End-to-end not validated | Not validated |
| A5 | Yes | Bring-up and direct/operator tests performed during this integration | `vllm serve` validation completed; full regression still required | Not validated |

Build support is not device acceptance. A3 must pass operator and model tests on
physical A3 hardware before it is marked accepted.

## Delivery stages

### Stage 1: cross-SoC eager ordinary inference

- A2/A3/A5 capability routing.
- BF16 eager prefill using Phase 6 when selected.
- Causal convolution and ordinary recurrent decode through FLA when selected.
- Native l2norm and existing model-level output processing.
- Six-stage legacy and diagnostic prefill composition.
- Cross-SoC unit, operator, and Qwen smoke coverage.

### Stage 1.1: close current fallback and strict-mode gaps

- Restart the six-stage pipeline safely after a non-state-mutating Phase 6 probe
  failure in `auto` mode.
- Validate only symbols required by the selected execution graph in strict mode.
- Make per-operator diagnostic mode explicit and test its interaction with the
  fused core.

These are future code changes. This document update does not implement them.

### Stage 2: complete A2/A3/A5 acceptance

- Run the FLA official matrix with the correct wheel on each SoC.
- Run vLLM-Ascend operator and model tests on physical hardware.
- Add performance and long-context comparisons.
- Record exact CANN, torch-npu, FLA commit, wheel, and OPP identities.

### Stage 3: eager MTP

- Remove the speculative-decode bypass only after accepted-token contracts are
  designed and tested.
- Validate full, partial, and rejected token acceptance.
- Verify convolution and SSM states commit only accepted tokens.

### Stage 4: ACL Graph without MTP

- Run all capability probes before capture.
- Validate graph-safe metadata and stable buffer addresses.
- Compare eager and graph outputs and state updates across capture buckets.

### Stage 5: MTP plus ACL Graph

- Validate target and drafter responsibilities separately.
- Validate accepted-token state transitions across graph replay.
- Extend to prefix caching, PCP/DCP, and broader distributed configurations only
  after the base combination is stable.

## Risks and mitigations

### Wrong SoC package

Mitigation: build and install per SoC, record loaded library and OPP identity,
and reject unsupported runtime signatures through probe failures.

### Fused call hides the failing internal stage

Mitigation: log the fused symbol and inputs, reproduce with the six-stage path,
and use FLA official tests/profiling to isolate internal failures.

### Fused and standalone metadata diverge

Mitigation: treat host `cu_seqlens` and `chunk_indices` as one validated pair,
cover dense and varlen cases, and compare final state as well as output.

### Runtime failure is incorrectly reported as fallback

Mitigation: distinguish resolution fallback from first-live-shape probe failure
in logs and reports. Do not claim whole-pipeline runtime fallback until the code
implements and tests it.

### Strict mode requires unused symbols

Mitigation: document the current full replacement-set requirement. A future
implementation may validate only the selected execution graph.

### Shared model path hides one model regression

Mitigation: keep one implementation but require separate Qwen3.5 and Qwen3.6
real-weight acceptance.

### Asynchronous state corruption

Mitigation: synchronize probes, clone stateful probe inputs, never blindly retry
after a possibly state-mutating failure, and log mutation risk.

## File map

| Purpose | Path |
| --- | --- |
| Model routing and cache ownership | `vllm_ascend/ops/gdn.py` |
| FLA adapter and dispatcher | `vllm_ascend/ops/gdn_fla.py` |
| Legacy compatibility import | `vllm_ascend/ops/gdn_a5.py` |
| Hardware capability mapping | `vllm_ascend/device/device_config.py` |
| Backend environment variables | `vllm_ascend/envs.py` |
| Hardware and adapter unit tests | `tests/ut/device/test_device_config.py`, `tests/ut/ops/test_gdn_fla.py` |
| Operator smoke tests | `tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py` |
| Qwen3.5 model smoke | `tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py` |
| Qwen3.6 model smoke | `tests/e2e/pull_request/two_card/test_qwen3_6_27b_fia.py` |
| A2/A3 Chinese operation guide | `docs/superpowers/guides/2026-08-29-qwen-gdn-a2-a3-validation-guide-zh.md` |

## Completion criteria

Stage 1 is accepted for a hardware family only when:

- the matching FLA wheel and OPP are installed and identified;
- the Phase 6 and recurrent public symbols resolve;
- official FLA operator tests pass on that hardware;
- vLLM-Ascend capability, adapter, and operator tests pass;
- Qwen3.5 and Qwen3.6 real-weight eager tests pass;
- native and FLA outputs/states meet the recorded numerical criteria;
- logs and profiling prove the intended symbols and kernels executed;
- unexpected fallback is explained rather than ignored;
- MTP or ACL Graph support is not advertised by this Stage 1 result.

Acceptance is recorded separately for A2, A3, and A5. Passing on one SoC does
not imply acceptance on another.
