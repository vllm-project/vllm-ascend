# Sequence Parallel Unification

## Status

This document defines the migration plan for unifying FlashComm1 and sequence
parallelism (SP) in vLLM Ascend. It is an implementation contract rather than a
description of the current user interface.

The target architecture keeps one user-facing SP feature and expresses its
distributed state explicitly in model-layer code. The existing compilation
passes remain available only while model families are migrated.

## Motivation

FlashComm1 and SP implement the same fundamental transformation:

```text
Tensor-parallel AllReduce
    -> ReduceScatter
    -> local normalization, quantization, or MoE computation
    -> AllGather only when a full-token activation is required
```

They currently use different control and execution paths:

| Area | FlashComm1 | Compilation-pass SP |
| --- | --- | --- |
| User control | `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `pass_config.enable_sp` |
| Implementation | Ascend linear layers, custom ops, and model patches | Inductor graph patterns |
| Execution mode | Eager and graph | Full-graph compilation |
| Token threshold | Forward-context policy | Compile-range policy |
| Sharding state | Explicit but distributed across call sites | Implicit in the transformed graph |

Maintaining both paths has introduced duplicated padding, communication, graph
size, PP-intermediate, and MoE dispatch logic. It also makes `enable_sp` mean
FlashComm1 in some modules and compilation-pass SP in others.

Upstream vLLM is moving model-level fusion from compiler passes to explicit
model code. The relevant upstream references are:

- [Changes in vLLM Model Development](https://github.com/vllm-project/vllm/issues/42770)
- [Porting compiler fusions to manual fusion](https://github.com/vllm-project/vllm/issues/43224)
- [Migrate MoE SP to a non-torch-compiled path](https://github.com/vllm-project/vllm/pull/47881)

The vLLM Ascend implementation should follow that direction instead of adding
new model-specific patterns to the SP passes.

## Goals

- Expose one sequence-parallel feature and one token threshold.
- Treat FlashComm1 kernels and delayed-AllGather strategies as Ascend SP
  optimizations, not as a separate feature.
- Represent whether an activation is full, tensor-parallel partial, or
  sequence-sharded explicitly.
- Support eager and graph execution through the same model-layer semantics.
- Preserve plain tensor parallelism as the fallback for unsupported or small
  batches.
- Migrate model families in independently reviewable changes.
- Remove the legacy SP passes after correctness and performance parity.

## Non-Goals

- Changing FlashComm2 or FlashComm3 semantics.
- Replacing context parallelism or its KV-cache sharding.
- Removing local compiler use for adjacent-operator fusion.
- Enabling an unsupported model or quantization scheme without model-specific
  correctness and performance evidence.

## Current Implementation Inventory

### Control Plane

- `vllm_ascend/envs.py` defines `VLLM_ASCEND_ENABLE_FLASHCOMM1`.
- `vllm_ascend/utils.py::enable_sp` resolves the FlashComm1 environment switch.
- `vllm_ascend/utils.py::enable_sp_by_pass` resolves compilation-pass SP.
- `vllm_ascend/platform.py` applies SP thresholds, graph sizes, and compatibility
  checks.
- `vllm_ascend/ascend_forward_context.py` carries the per-forward FlashComm1
  decision, token padding, and fusion policy.
- `vllm_ascend/worker/model_runner_v1.py` pads token counts and adapts PP
  intermediate tensors for both implementations.

### FlashComm1 Data Plane

- `vllm_ascend/ops/linear_op.py::SequenceRowParallelOp` changes a row-parallel
  reduction from AllReduce to ReduceScatter and selects fused Matmul-ReduceScatter
  kernels where available.
- `vllm_ascend/ops/linear_op.py::SequenceColumnParallelOp` gathers a
  sequence-sharded activation before a column-parallel projection.
- `vllm_ascend/ops/register_custom_ops.py` implements conditional gather,
  padding, unpadding, and reduction custom ops.
- `vllm_ascend/ops/fused_moe/prepare_finalize.py` adapts MoE dispatch and combine
  for sequence-sharded input.
- Model patches handle structures that cannot infer the sharding transition from
  a generic linear prefix, including VL deepstack inputs and hybrid attention.

### Compilation-Pass Data Plane

- `SequenceParallelismPass` rewrites AllReduce plus RMSNorm patterns into
  ReduceScatter plus local RMSNorm plus AllGather.
- `SequenceParallelismMoePass` moves or removes AllGather operations around MoE
  layers and sequence chunking.
- Both passes rely on the exact FX graph and currently require graph execution.

## Target Terminology

The user-facing feature is named **sequence parallelism**. FlashComm1 becomes an
implementation detail of the Ascend SP backend.

The following state names are used throughout the implementation:

| State | Meaning |
| --- | --- |
| `FULL` | Every TP rank owns the full activation. |
| `TP_PARTIAL` | Every TP rank owns a partial sum that still requires reduction. |
| `SEQUENCE_SHARDED` | Tokens are partitioned over the TP group and each local value is reduced. |

The state describes activation ownership, not whether a particular optimized
kernel was used.

## Target Control Plane

The final control plane has two values:

- An enable decision for sequence parallelism.
- A minimum token threshold at which a forward changes from plain TP to SP.

During migration, the canonical values continue to use upstream-compatible
`pass_config.enable_sp` and `pass_config.sp_min_token_num`. They must be copied
into an Ascend runtime policy before model construction; model code must not
read or mutate `pass_config` directly.

`VLLM_ASCEND_ENABLE_FLASHCOMM1` remains a deprecated compatibility alias during
the transition. Conflicting canonical and legacy values must fail with a clear
configuration error instead of silently choosing one.

The runtime decision must satisfy all of the following:

```text
configured
and tensor_parallel_size > 1
and model/backend capability is supported
and num_tokens >= min_token_threshold
```

Unsupported configurations use plain TP unless the user explicitly requested a
strict mode in a future interface.

## Target Data Plane

### Layer Contract

Row-parallel projections expose their unreduced output by setting
`reduce_results=False`. A shared Ascend SP component then performs one of the
following transitions:

```text
TP_PARTIAL -> FULL
    AllReduce

TP_PARTIAL -> SEQUENCE_SHARDED
    optional padding -> ReduceScatter

SEQUENCE_SHARDED -> FULL
    AllGather -> optional unpadding
```

Normalization and quantization operate on the local sequence shard where the
model contract permits it. AllGather is delayed until an attention projection,
output boundary, or model-specific consumer requires full tokens.

### Model-Specific Transitions

Model code or a pluggable Ascend layer must explicitly handle transitions for:

- VL embedding and deepstack additions.
- MLA projections where AllGather can be delayed beyond QKV projection.
- MoE routing, dispatch, combine, and shared experts.
- MTP and auxiliary hidden-state consumers.
- PP intermediate tensors.
- Hybrid attention and recurrent layers.

Prefix matching alone must not be the source of truth for these transitions.

### Execution Modes

The same layer semantics must run in eager and graph modes. Graph capture may
select optimized kernels, but it must not change activation ownership or insert
the fundamental SP transformation through a model-specific FX pattern.

## Migration Plan

### Phase 1: Inventory and Contract

- Land this design and the migration matrix.
- Freeze new model-specific patterns in both SP passes.
- Identify current tests and missing coverage for each model family.

Exit criteria:

- Review agreement on state names, configuration precedence, and pass-removal
  criteria.

### Phase 2: Unified Configuration and State

- Add an immutable Ascend SP policy derived from `VllmConfig`.
- Add the activation-state type and transition helpers.
- Route existing FlashComm1 decisions through the policy without changing
  kernels or model behavior.
- Add compatibility handling for legacy environment variables.

Exit criteria:

- Existing FlashComm1 and pass configurations resolve deterministically.
- Unit tests cover TP size, token thresholds, conflicting switches, and eager
  mode.

### Phase 3: Dense Models

- Move the baseline AllReduce-to-ReduceScatter transformation into explicit
  dense decoder-layer code.
- Cover unquantized and supported quantized normalization paths.
- Verify the first layer, last layer, and output unpadding boundaries.

Exit criteria:

- Dense eager and graph outputs match plain TP baselines.
- Performance is no worse than the current FlashComm1/pass path for accepted
  benchmark variance.

### Phase 4: MoE Models

- Unify MoE sequence sharding with the existing dispatch/combine abstraction.
- Integrate delayed AllGather and dynamic-quant behavior as SP backend policies.
- Cover shared experts and expert parallelism.

Exit criteria:

- Correctness for supported All2All backends and TP/EP combinations.
- No duplicate token routing or redundant AllGather plus chunk pairs.

### Phase 5: VL and MLA Models

- Move deepstack chunking and embedding-boundary behavior into explicit model
  adapters.
- Migrate MLA-specific delayed-AllGather behavior.
- Remove the corresponding graph patterns after each family reaches parity.

Exit criteria:

- VL prompt embeddings, multimodal inputs, and MLA paths pass eager and graph
  tests.

### Phase 6: MTP, PP, LoRA, and CP Compatibility

- Make MTP and PP boundaries declare and restore activation ownership.
- Validate auxiliary hidden states, draft models, and LoRA consumers.
- Validate coexistence rules with PCP and DCP.

Exit criteria:

- The compatibility matrix below is complete for all supported combinations.

### Phase 7: Pass Removal

- Disable legacy SP passes by default after all supported model families migrate.
- Remove pass registration, patterns, pass-only custom ops, and configuration
  workarounds.
- Remove the FlashComm1 environment alias after its deprecation window.

Exit criteria:

- No supported configuration requires either SP pass.
- Release notes and the user guide describe one SP feature.

## Validation Matrix

Every migrated model family must record the applicable cells below. A cell may
be marked unsupported only with an explicit reason.

| Dimension | Required cases |
| --- | --- |
| Execution | eager, graph |
| Model | Dense, MoE, VL Dense, VL MoE, MLA |
| Precision | BF16/FP16, each supported activation quantization path |
| Parallelism | TP, TP+EP, TP+PP, TP+PCP/DCP where supported |
| Features | MTP, LoRA, prompt embeddings, auxiliary hidden states |
| Workload | below threshold, at threshold, above threshold, non-TP-divisible token count |

Validation has three levels:

1. Unit tests for policy and state transitions without NPU hardware where
   possible.
2. Multi-card correctness tests comparing SP with plain TP.
3. NPU performance tests measuring throughput, latency, memory, and startup time.

## Rollback Strategy

Each model-family migration must be independently revertible. Until pass removal,
the legacy implementation may be selected internally for an unmigrated family,
but one forward must never run both the manual and pass transformations.

If a migrated family fails its performance gate, retain the explicit state
contract and fall back to unfused communication operators while the optimized
kernel is corrected. Do not restore model-specific FX patterns as the default
solution.

## Commit Strategy

Implementation changes are split into reviewable commits in this order:

1. Design and inventory.
2. Configuration policy with unit tests.
3. Activation-state helpers with unit tests.
4. Existing FlashComm1 call sites routed through the policy.
5. One commit series per model family and compatibility feature.
6. Pass and compatibility-code removal only after validation is complete.

No commit should combine a new state abstraction, multiple model migrations,
and pass deletion.
