# Qwen3.5/Qwen3.6 GDN A5 Operator Integration Design

## Status

- Date: 2026-08-25
- Repository: `vllm-ascend`
- Target: Ascend A5 / Ascend 950
- Models: Qwen3.5 and Qwen3.6
- Dependency: `flash-linear-attention-npu` installed as `fla_npu`
- Delivery strategy: eager inference first, then MTP, ACL Graph, and their combination

## Summary

Add an A5-specific GDN operator adapter to `vllm-ascend`. The adapter selects
between operators exposed by `flash-linear-attention-npu` and the existing
vLLM-Ascend implementations. It performs capability probes before serving
requests, logs the exact implementation selected for every logical operator,
and falls back to the existing implementation when a replacement cannot be
loaded or does not pass its probe.

The initial eager inference path has nine dispatch points:

1. `causal_conv1d`
2. `l2norm_fwd`
3. `chunk_local_cumsum`
4. `chunk_scaled_dot_kkt`
5. `solve_tri`
6. `recompute_w_u_fwd`
7. `chunk_gated_delta_rule_fwd_h`
8. `chunk_fwd_o`
9. `recurrent_gated_delta_rule`

Qwen3.5 and Qwen3.6 use the same upstream Qwen3.5 model implementation and the
same `QwenGatedDeltaNetAttention` patch. They therefore share one operator
adapter, while retaining separate end-to-end model tests.

Only `vllm-ascend` is changed. Neither upstream `vllm` nor
`flash-linear-attention-npu` is modified.

## Goals

- Use the available `fla_npu` GDN forward operators on A5.
- Preserve the current vLLM-Ascend implementation as a per-operator fallback.
- Make backend selection observable and make failures attributable to one
  logical operator and one concrete implementation.
- Support ordinary prefill and single-token decode in eager mode first.
- Reach MTP plus ACL Graph through separately validated stages.
- Keep behavior on non-A5 devices unchanged.

## Non-goals for the first stage

- Training or backward propagation.
- MTP/speculative decode.
- ACL Graph capture or replay.
- Prefix caching.
- PCP or DCP.
- Broad changes to vLLM scheduling or model definitions.
- Modifying or repackaging `flash-linear-attention-npu`.

## Existing implementation

The upstream vLLM tree already contains the Qwen3.5 GDN model layer and GDN
attention metadata. Qwen3.6 reuses these Qwen3.5 classes rather than having an
independent Qwen3.6 Python model file.

The existing vLLM-Ascend path is centered in:

- `vllm_ascend/ops/gdn.py`
- `vllm_ascend/ops/gdn_attn_builder.py`
- `vllm_ascend/patch/worker/patch_qwen3_5.py`

`gdn.py` currently performs projection preparation, causal convolution, gate
generation, metadata/cache handling, prefill/decode separation, chunk GDN
prefill, recurrent GDN decode, and result merging. The Qwen3.5 patch targets the
shared `QwenGatedDeltaNetAttention`, so it also covers Qwen3.6.

`flash-linear-attention-npu/examples/flash_gated_delta_rule.py` demonstrates
the required chunk GDN forward composition. The installed package exposes
stable public imports under `fla_npu.ops.ascendc` and
`fla_npu.ops.triton`; the example file itself is not imported at runtime.

## Architecture

### A5 adapter

Add `vllm_ascend/ops/gdn_a5.py` with three responsibilities:

1. Resolve and probe operator implementations.
2. Normalize layouts, dtypes, keyword arguments, and return values.
3. Execute the selected prefill or decode operator chain.

The module exposes stable vLLM-Ascend-facing functions such as:

```text
gdn_causal_conv1d(...) -> conv_output
gdn_prefill(...) -> output, final_state
gdn_decode(...) -> output
```

It does not know model names, scheduler behavior, or how model layers are
patched. It receives prepared tensors and metadata from `gdn.py`.

### Existing GDN layer

`vllm_ascend/ops/gdn.py` remains responsible for:

- projection output preparation;
- deciding prefill, ordinary decode, speculative decode, or mixed-batch paths;
- obtaining attention metadata and cache indices;
- gathering and scattering convolution and SSM states;
- merging outputs back into vLLM token order;
- calling the A5 adapter only on supported A5 configurations.

Non-A5 hardware follows the existing code path without importing `fla_npu`.

### Model patch and metadata

The shared Qwen3.5 patch remains the model entry point for both Qwen3.5 and
Qwen3.6. No model-specific Qwen3.6 branch is added.

`gdn_attn_builder.py` remains unchanged in the first eager stage. Later stages
may add graph-stable metadata only when ACL Graph requirements are known.

## Operator registry

The A5 adapter has a registry keyed by logical operator name. Each entry owns:

- the preferred `fla_npu` resolver;
- the current vLLM-Ascend resolver;
- a capability predicate;
- a scratch-tensor smoke probe;
- whether the operator can mutate persistent state;
- a normalized call wrapper;
- a cached selection for each supported runtime signature.

| Logical operator | Preferred replacement | Existing fallback |
| --- | --- | --- |
| `causal_conv1d` | `fla_npu.ops.ascendc.causal_conv1d` | `torch.ops._C_ascend.npu_causal_conv1d_custom` |
| `l2norm_fwd` | `fla_npu.ops.triton.l2norm_fwd` | vLLM FLA `l2norm_fwd` |
| `chunk_local_cumsum` | `fla_npu.ops.triton.chunk_local_cumsum` | vLLM-Ascend Triton implementation |
| `chunk_scaled_dot_kkt` | compatible `fla_npu` AscendC or Triton API | vLLM-Ascend Triton implementation |
| `solve_tri` | `fla_npu.ops.ascendc.solve_tri`, then its Triton API | vLLM-Ascend `solve_tril` |
| `recompute_w_u_fwd` | `fla_npu.ops.ascendc.recompute_w_u_fwd` | vLLM-Ascend Triton implementation |
| `chunk_gated_delta_rule_fwd_h` | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h` | `torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h` |
| `chunk_fwd_o` | `fla_npu.ops.ascendc.chunk_fwd_o` | `torch.ops._C_ascend.chunk_fwd_o` |
| `recurrent_gated_delta_rule` | available installed recurrent API | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` |

The README test labels `gdn_fwd_h`, `gdn_fwd_o`, and `recompute_wu_fwd` map to
the public Python names `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`, and
`recompute_w_u_fwd`.

### Recurrent resolver order

The installed recurrent interface is not currently exported consistently by
all `fla_npu` A5 packages. Its resolver tries these interfaces in order and
accepts the first one that passes the current stage's probe:

1. `fla_npu.ops.ascendc.recurrent_gated_delta_rule`, if exported;
2. `torch_npu.npu_recurrent_gated_delta_rule`, if registered by the installed
   package/OPP;
3. `torch.ops.ascend_ops.recurrent_gated_delta_rule`, if its extension is
   already loaded;
4. `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` as the existing
   vLLM-Ascend fallback.

The first eager stage probes only ordinary decode. MTP accepted-token support
and ACL Graph support are separate capabilities and are not inferred from an
ordinary decode success.

### Backward operators

The following available operators are not connected to vLLM inference:

- `prepare_wy_repr_bwd_full`
- `chunk_gated_delta_rule_bwd_dhu`
- `chunk_bwd_dv_local`
- `prepare_wy_repr_bwd_da`
- `chunk_bwd_dqkwg`

They may be reported by an optional diagnostic capability dump, but they are
not imported, probed, or executed during normal inference startup. Adding them
would create an unsupported training surface without helping serving.

## Backend configuration

Configuration is centralized in `vllm_ascend/envs.py`, in accordance with the
repository environment-variable rules.

### Global mode

`VLLM_ASCEND_GDN_BACKEND` accepts:

- `auto`: on A5, prefer a replacement that passes its probe and otherwise use
  the existing implementation; this is the A5 default;
- `fla_npu`: strict validation mode; a required replacement failure stops model
  initialization;
- `native`: use only the existing vLLM-Ascend implementations.

On non-A5 hardware the effective default is `native`.

### Per-operator override

`VLLM_ASCEND_GDN_OP_BACKENDS` is an optional diagnostic override using
comma-separated `operator=backend` entries, for example:

```text
causal_conv1d=fla_npu,chunk_fwd_o=native
```

It is used for per-operator accuracy isolation and incident diagnosis. Invalid
operator names, duplicate entries, or invalid backend values fail during
configuration validation rather than being ignored.

Both variables require normal code review under the repository's environment
variable policy.

## Capability probing and smoke tests

There are two distinct smoke-test layers.

### Runtime operator smoke probes

Runtime probes select safe implementations; they are not a replacement for
repository tests.

Probes run after the NPU device and CANN runtime are initialized, before the
first user request and before ACL Graph capture. They use isolated scratch
tensors and call `torch.npu.synchronize()` after each operator so asynchronous
launch or ABI errors are attributed to the correct operator.

Each probe verifies the properties relevant to that operator:

- import and public symbol resolution;
- OPP/op-api library loading;
- supported dtype and layout;
- representative Qwen head dimensions;
- output shape, dtype, and finite values;
- state update semantics for causal convolution and recurrent GDN;
- grouped `Nk`/`Nv` handling where applicable;
- required optional arguments and return tuple shape.

Stateful probes clone scratch convolution or SSM states, verify the expected
mutation, and discard the buffers. They never touch a model cache.

Probe results are cached by a runtime signature containing at least:

```text
SoC, tensor dtype, state dtype, Nk, Nv, Dk, Dv, chunk size,
ordinary/MTP mode, eager/ACL Graph mode
```

This prevents a BF16 success from being incorrectly reused for FP32 state, or
an ordinary decode success from being treated as proof of MTP support.

### Repository-level A5 smoke tests

Add explicit A5 tests under `tests/`:

1. Unit tests for resolver order, configuration parsing, cached selections,
   fallback decisions, and log fields. These tests use mocks and do not require
   an NPU.
2. A single-card A5 operator-chain smoke test using small tensors. It forces
   each replacement independently, compares it with the native path, then runs
   the full replacement chain.
3. A Qwen3.5 real-weight eager smoke test based on the existing 0.8B test.
4. A Qwen3.6 real-weight eager smoke test using the existing supported
   Qwen3.6 configuration.

The operator-chain smoke test covers:

- prefill lengths 1, 63, 64, and 65 to cross chunk boundaries;
- one and multiple sequences;
- ordinary single-token decode;
- a mixed batch containing decode and prefill work;
- convolution and SSM cache updates across consecutive calls;
- `auto`, strict `fla_npu`, and `native` modes;
- simulated import, symbol, and probe failures;
- backend-selection and execution-error logs.

The end-to-end model smoke tests use deterministic greedy decoding, require a
non-empty output, compare replacement and native generated tokens, and impose
a timeout so a hang becomes a test failure.

## Data flow

### Causal convolution

`gdn.py` prepares the existing convolution metadata and cache references. The
A5 adapter calls the selected normalized causal convolution wrapper with:

- prefill/decode run mode;
- query start locations;
- cache indices and initial-state mode;
- convolution state buffer;
- accepted-token metadata only in the later MTP stage.

Both replacement and fallback wrappers must produce the same flattened packed
QKV representation and the same cache mutation semantics.

### Q/K normalization

The selected `l2norm_fwd` normalizes Q and K before both chunk prefill and
recurrent decode. Normalization happens once. Repeating normalized grouped
heads is numerically equivalent to normalizing after repeating but avoids
redundant work.

### Prefill

After convolution and gate preparation, tensors have these logical layouts:

```text
q, k: [1, T, Nk, Dk]
v:    [1, T, Nv, Dv]
g:    [1, T, Nv], FP32
beta: [1, T, Nv]
```

When `Nv > Nk`, Q and K are repeated by `Nv / Nk` for the chunk forward
operators, matching the reference example. The adapter validates exact
divisibility before repeating.

The prefill composition is:

```text
l2norm_fwd(q, k)
-> chunk_local_cumsum(g)
-> chunk_scaled_dot_kkt(k, beta, cumulative_g)
-> solve_tri(A)
-> recompute_w_u_fwd(k, v, beta, A, cumulative_g)
-> chunk_gated_delta_rule_fwd_h(k, w, u, cumulative_g, initial_state)
-> chunk_fwd_o(q, k, new_v, h, cumulative_g)
```

The vLLM SSM cache is stored as:

```text
[state_block, Nv, Dv, Dk]
```

The chunk GDN composition consumes and returns:

```text
[sequence, Nv, Dk, Dv]
```

The adapter gathers the requested blocks, clears sequences without an initial
state, transposes to chunk layout, executes the operator chain, transposes the
final state back, casts to the configured cache dtype, and scatters it into the
original cache blocks.

Existing prebuilt variable-length/chunk metadata is reused where compatible.
The adapter may derive only the missing package-specific representation; it
must not perform device-to-host `.item()` calls in the hot path.

### Ordinary decode

Decode uses:

```text
q, k:              [T, Nk, Dk]
v:                 [T, Nv, Dv]
g, beta:           [T, Nv]
state:             [state_block, Nv, Dv, Dk]
actual_seq_lengths
ssm_state_indices
```

The selected recurrent wrapper normalizes differing API keyword names and
return conventions. It must update the selected SSM cache blocks in place and
return `[T, Nv, Dv]` output.

The mixed-batch path retains the current order: peel ordinary decode tokens,
execute recurrent decode, execute chunk prefill, and concatenate results back
in decode-first order.

## Fallback and failure semantics

Fallback occurs only during resolution or pre-request probing:

- package import failure;
- OPP/op-api loading failure;
- missing symbol;
- incompatible signature;
- unsupported runtime signature;
- smoke-probe failure.

`auto` records the reason and selects the existing implementation.
`fla_npu` raises a model-initialization error containing the missing or failed
operator list. `native` does not import `fla_npu`.

An operator execution failure during a real request is not automatically
retried. This is mandatory for stateful causal convolution and recurrent GDN,
because an asynchronous failure may occur after a cache was partially
modified. Retrying could apply the same tokens twice. Runtime failures are
logged and propagated so the request or worker fails visibly.

Individual operators may be mixed between `fla_npu` and native only through
normalized wrappers whose inputs and outputs have been verified equivalent.
If an intermediate contract cannot be normalized safely, the entire chunk
prefill composition is selected as one backend unit rather than mixing unsafe
intermediates.

## Logging and diagnostics

Backend selection is logged once per runtime signature, not once per token or
layer invocation.

An information log contains:

- logical operator;
- selected backend (`fla_npu` or `native`);
- exact Python/dispatcher symbol;
- runtime signature;
- supported feature level: ordinary, MTP, and ACL Graph.

Example:

```text
GDN A5 operator selected: op=chunk_fwd_o backend=fla_npu
symbol=fla_npu.ops.ascendc.chunk_fwd_o dtype=bf16 dk=128 dv=128
mode=eager capability=ordinary
```

A fallback warning contains the requested and selected backends, probe stage,
exception type, and concise reason:

```text
GDN A5 operator fallback: op=solve_tri requested=fla_npu selected=native
stage=smoke_probe reason="installed symbol rejected the requested chunk size"
```

A runtime error contains:

- model layer name;
- prefill/decode/MTP phase;
- logical operator and concrete symbol;
- backend;
- input shapes, dtypes, device, and contiguity;
- chunk size and sequence count where relevant;
- whether persistent state may have been mutated;
- original exception type and message.

Tensor values are not logged. Shape diagnostics must not introduce device
synchronization in the normal path.

## Testing and acceptance criteria

### Unit tests

Unit tests cover:

- lazy import and resolver priority;
- global and per-operator configuration validation;
- A5 versus non-A5 selection;
- probe-result caching by runtime signature;
- strict-mode initialization failure;
- `auto` fallback for each failure class;
- no import of `fla_npu` in `native` mode;
- normalized state layouts and grouped heads;
- required structured log fields;
- no runtime retry after an execution failure.

### A5 operator-chain tests

For each of the nine dispatch points:

1. force the replacement while all other operators remain native;
2. compare output and relevant state with the all-native baseline;
3. force a probe failure and confirm native fallback;
4. confirm the selection log names the actual implementation.

After isolated validation, run the complete replacement chain.

Initial numerical criteria are:

- output cosine similarity at least `0.999`;
- output `rtol=5e-3`, `atol=5e-3` for BF16 paths;
- state compared separately with dtype-appropriate tolerances;
- all outputs and states finite;
- deterministic greedy model tokens equal to native on the smoke prompt set.

A tolerance adjustment requires recorded evidence from the reference/native
comparison; tests must not silently widen thresholds.

### Model coverage

Qwen3.5 and Qwen3.6 must both run real-weight eager smoke tests even though
they share the same Python implementation. Passing only one model is not
sufficient acceptance evidence.

## Delivery stages

### Stage 1: eager ordinary prefill and decode

- Add the adapter and registry.
- Add all nine dispatch points.
- Add capability probes, fallback, and logging.
- Add unit, operator-chain, and Qwen3.5/Qwen3.6 smoke tests.
- Validate TP1; include TP2 where model size or the existing test requires it.

### Stage 2: eager MTP

- Probe recurrent `num_accepted_tokens` support separately.
- Validate full acceptance, partial acceptance, and rejection.
- Verify convolution and SSM states commit only accepted tokens.
- Add explicit test timeouts to detect hangs.

### Stage 3: ACL Graph without MTP

- Run all probes before capture.
- Validate graph capture/replay for supported batch buckets.
- Add functional/non-mutating wrappers only where graph capture requires them.
- Compare eager and graph results and cache state.

### Stage 4: target graph plus eager MTP drafter

- Use `FULL_DECODE_ONLY` for the target model.
- Keep the MTP drafter eager initially.
- Validate accepted-token state transitions across graph replays.

### Stage 5: final combinations

- MTP plus ACL Graph.
- Prefix cache.
- TP/EP combinations.
- PCP and DCP where supported.
- Long-running stability and performance regression tests.

## Planned file changes

- Add `vllm_ascend/ops/gdn_a5.py`.
- Update `vllm_ascend/ops/gdn.py` to route A5 operations through the adapter.
- Update `vllm_ascend/envs.py` for reviewed backend configuration.
- Reuse `vllm_ascend/ops/gdn_attn_builder.py` in Stage 1; change it only when
  later graph metadata requires it.
- Reuse `vllm_ascend/patch/worker/patch_qwen3_5.py` for both model families.
- Add focused unit tests under `tests/ut/ops/`.
- Add A5 operator smoke tests under `tests/e2e/nightly/single_node/ops/`.
- Extend existing Qwen3.5 and Qwen3.6 model tests or add focused A5 variants.

## Risks and mitigations

### Package surface differs by build

Mitigation: resolve public symbols lazily, probe concrete capabilities, and log
the concrete symbol instead of assuming a package version implies support.

### Recurrent A5 export is incomplete

Mitigation: treat recurrent APIs as ordered candidates and retain the current
vLLM-Ascend recurrent GDN as the final fallback.

### Layout or state-dtype mismatch

Mitigation: centralize transformations in the adapter, probe each state dtype,
and compare final state as well as attention output.

### Runtime fallback corrupts cache

Mitigation: fallback only before serving; never retry a failed stateful
operator on a live request.

### Per-operator logging harms decode performance

Mitigation: log selection once per runtime signature and emit per-call logs
only on failure.

### Shared model code hides Qwen3.6 regressions

Mitigation: retain one implementation but require independent real-weight
Qwen3.5 and Qwen3.6 smoke tests.

## Completion criteria

Stage 1 is complete only when:

- all nine dispatch points are implemented;
- every selected backend is visible in logs;
- missing or incompatible replacements fall back correctly in `auto` mode;
- strict mode reports precise initialization failures;
- isolated and complete-chain A5 smoke tests pass;
- Qwen3.5 and Qwen3.6 eager real-weight smoke tests pass;
- native behavior on non-A5 hardware remains unchanged;
- no unsupported MTP or ACL Graph capability is advertised.
