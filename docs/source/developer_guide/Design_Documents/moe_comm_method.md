# MoE Communication Method Selection

## Overview

`select_moe_comm_method(...)` chooses the communication implementation used by
Ascend MoE layers. The selector is intentionally kept as a small policy layer:
it reads runtime configuration, checks whether MoE expert parallelism is active,
dispatches to a hardware-specific helper, and records the selected method in
the forward context.

The selector does not implement the communication kernels themselves. The
selected `MoECommType` is later mapped to a concrete implementation in
`vllm_ascend/ops/fused_moe/moe_comm_method.py`.

## Communication Types

| Type | Meaning |
| --- | --- |
| `ALLGATHER` | Gather routed tokens from all expert-parallel ranks, run local expert compute, then reduce as needed. This is the conservative fallback for several devices and disabled expert parallelism. |
| `ALLTOALL` | Exchange routed tokens across expert-parallel ranks through all-to-all style token dispatch and combine. |
| `MC2` | Use the regular MC2 dispatch/combine path, bounded by `get_mc2_tokens_capacity()`. |
| `FUSED_MC2` | Use a fused MoE communication/operator path. The concrete fused operator depends on `enable_fused_mc2` and hardware capability. |

## Selection Inputs

The public selector uses the following inputs:

- whether the current model is an MoE model,
- whether expert parallelism is effectively enabled,
- Ascend device generation,
- current `num_tokens`,
- regular MC2 token capacity,
- MoE quantization type,
- `enable_fused_mc2` and fused-operator-specific capacity.

The selector deliberately does not branch on prefill/decode phase, `kv_role`,
`is_kv_producer`, or `is_kv_consumer`. PD-disaggregated execution and splitfuse
shape the runtime context and scheduling above this layer; they should not make
the MoE communication selector split into phase-specific policy.

## Hardware Policy

### Common gating

Non-MoE models return `None`.

If expert parallelism is disabled, or the EP group has world size 1, the
selector returns `ALLGATHER`. Hardware-specific selection starts only after this
common gate passes.

### A2

A2 selects regular `MC2` only when all regular MC2 conditions pass:

- the number of experts per device is at most 24,
- EP world size is at least 16,
- `num_tokens` fits `get_mc2_tokens_capacity()`.

Otherwise A2 falls back to `ALLGATHER`.

### A3

A3 has the most complex policy because it supports regular MC2 and two fused
MC2-like paths.

`enable_fused_mc2=0` disables fused MC2 selection. A3 then uses regular `MC2`
when `num_tokens` fits regular MC2 capacity, otherwise `ALLTOALL`.

`enable_fused_mc2=1` selects the legacy `dispatch_ffn_combine` path when:

- EP world size is at most 32,
- `num_tokens` fits `mega_moe_max_tokens`.

This path is treated as legacy and isolated in `_can_use_dispatch_ffn_combine`
and `_fits_dispatch_ffn_combine_capacity`. Its capability checks do not affect
`enable_fused_mc2=2` or regular MC2.

`enable_fused_mc2=2` selects `dispatch_gmm_combine_decode` when:

- the speculative/MTP helper allows the path,
- MoE quantization is `w8a8_dynamic`,
- `num_tokens` fits the `dispatch_gmm_combine_decode` batch-size capacity.

The current capacity used by the selector is 256 tokens, matching the current
C++ tiling `MAX_BATCH_SIZE` guard for `dispatch_gmm_combine_decode`.

If no fused path is selected, A3 falls back to regular `MC2` within regular MC2
capacity, otherwise `ALLTOALL`.

### A5

A5 selects regular `MC2` when `num_tokens` fits regular MC2 capacity and the
parallel world size is greater than 1.

If MC2 is not selected, A5 falls back to:

- `ALLGATHER` when world size is less than or equal to the number of top-k
  experts per token,
- otherwise `ALLTOALL`.

### 310P

310P currently uses `ALLGATHER`.

## Capacity Boundaries

Regular MC2 and fused MC2 do not share a single capacity source.

| Path | Capacity source |
| --- | --- |
| Regular `MC2` | `get_mc2_tokens_capacity()` |
| `enable_fused_mc2=1` / `dispatch_ffn_combine` | `get_ascend_config().mega_moe_max_tokens` |
| `enable_fused_mc2=2` / `dispatch_gmm_combine_decode` | current operator batch-size capacity, 256 tokens |

Keeping these capacities separate is important. Regular MC2 buffer sizing is
also used by profile runs and `TokenDispatcherWithMC2` global batch sizing, so
fused operator limits must not overwrite regular MC2 capacity.

## Configuration Meaning

`enable_fused_mc2` can be set through additional config, or through
`VLLM_ASCEND_ENABLE_FUSED_MC2` during the migration period.

| Value | Meaning |
| --- | --- |
| `0` | Do not select fused MC2. Use regular hardware policy. |
| `1` | Allow legacy `dispatch_ffn_combine` on A3 when its capability and capacity checks pass. |
| `2` | Allow `dispatch_gmm_combine_decode` on A3 when its capability, quantization, and capacity checks pass. |

`mega_moe_max_tokens` controls the per-rank token capacity for the
`dispatch_ffn_combine` path. Raising it increases workspace memory use.

## Extension Rules

When adding a new hardware generation or fused MoE path:

1. Keep `select_moe_comm_method(...)` as the public entry point.
2. Add or update a hardware-specific helper instead of growing the public
   selector body.
3. Keep operator capability checks in `_can_use_*` helpers.
4. Keep capacity checks in `_fits_*` helpers.
5. Do not reuse regular MC2 capacity for fused operators unless that capacity
   source is explicitly valid for the fused operator.
6. Keep logging at the public selector exit so branch helpers do not duplicate
   log messages.

## Verification Limits

Mock unit tests can prove that the selector returns the expected
`MoECommType` for a given configuration matrix. They cannot prove HCCL behavior,
kernel correctness, or performance on real NPUs.

For changes that alter hardware policy or fused capacity, PR validation should
include targeted NPU E2E, benchmark, or CI coverage. If a local environment
cannot cover A2, A3, A5, 310P, PD, or splitfuse scenarios, state that limitation
in the PR test section.

## Related Files

- `vllm_ascend/ascend_forward_context.py`, selector entry point, hardware
  helpers, and capacity helpers.
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`, mapping from `MoECommType`
  to concrete communication implementations.
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`, regular MC2 token dispatch
  and `global_bs` sizing.
- `vllm_ascend/envs.py`, migration-period environment variable definition for
  `VLLM_ASCEND_ENABLE_FUSED_MC2`.
