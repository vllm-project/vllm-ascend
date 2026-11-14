# MegaMoeLowLatency integration and Shared API v6

This document describes how `MegaMoeLowLatency` is integrated into
vLLM Ascend, why the low-latency communication path differs from the generic
MoE AllGather path, and how the Shared API v6 extension folds shared experts
into the operator.

## Goals

The integration targets decode-heavy MoE workloads on Ascend 950. It keeps
collectives in vLLM Ascend and lets the operator own the work that must stay
coupled with expert execution:

- gate matrix multiplication;
- global TopK routing;
- local routed-expert GMM1, activation, and GMM2;
- local result combination;
- optionally, the shared expert through Shared API v6.

The resulting distributed path is:

```text
AllGather(hidden states)
        |
        v
MegaMoeLowLatency / MegaMoeLowLatencyWithShared
        |
        v
ReduceScatter(output)
```

The feature is opt-in. Unsupported configurations and forward shapes retain
the regular AllGather MoE implementation.

## Why the generic preparation path was inefficient

The generic AllGather MoE path computes routing outside the fused operator and
prepares two tensors independently:

```text
AddRmsNorm
  -> Gate MatMul
  -> prepare(hidden states, router logits)
       -> AllGather(hidden states)
       -> AllGather(router logits)
  -> expert dispatch and computation
  -> ReduceScatter
```

The first `MegaMoeLowLatency` integration already passed the gate weight to
the operator, so the operator recomputed routing internally. To preserve the
generic `prepare()` interface, it replaced the full router-logit tensor with
`hidden_states[:, :1]`. This reduced the second collective's payload but did
not remove the work:

```text
AddRmsNorm             [local_tokens, hidden_size]
Gate MatMul            [local_tokens, num_experts]  # not consumed by the operator
Slice                  [local_tokens, 1]            # dummy router tensor
AllGather(hidden)      local_tokens -> global_tokens
AllGather(router info) local_tokens -> global_tokens
MegaMoeLowLatency      [global_tokens, hidden_size]
ReduceScatter          global_tokens -> local_tokens
```

There were therefore three redundant operations:

1. The external Gate MatMul produced logits that the operator did not use.
2. The Slice only constructed a placeholder for the generic interface.
3. The second AllGather communicated that unused placeholder.

## Routing-free preparation

`PrepareAndFinalizeWithAllGather.prepare_low_latency()` gathers only BF16
hidden states. It preserves the existing EP/SP, DP, and PCP padding and
collective rules, but it does not accept or create router logits.

`AscendRoutedExperts.forward_impl()` selects this path before generic MoE
preparation. It passes the gathered hidden states, gate weight, router
configuration, routed-expert weights, and optional shared-expert payload to
the operator. The result is finalized by the existing ReduceScatter path.

The new execution chain is:

```text
AddRmsNorm
  -> AllGather(hidden states)
  -> MegaMoeLowLatency
       -> Gate MatMul
       -> global TopK
       -> local routed experts
       -> local combine
  -> ReduceScatter
```

For Shared API v6, the shared MLP is part of the same operator call:

```text
AddRmsNorm
  -> AllGather(hidden states)
  -> MegaMoeLowLatencyWithShared
       -> routing and routed experts
       -> shared gate and shared MLP
       -> routed + shared local result
  -> ReduceScatter
```

This is valid because every rank receives all attention-output tokens, runs
the experts stored on that rank, and contributes its partial result. The final
ReduceScatter both combines rank-local expert contributions and returns the
token shard owned by each rank.

## Skipping the standalone gate and shared-expert graph

Removing the second AllGather inside `AscendRoutedExperts` is not sufficient:
the outer `AscendMoERunner` previously computed router logits before entering
the routed-expert custom operation and scheduled the shared MLP separately.

The runner now determines `mega_moe_low_latency_shared_active` before entering
the custom operation. When it is active:

- `_prepare_router_and_milestones()` is not called, so the standalone routed
  Gate MatMul is skipped;
- the standalone shared-expert preparation and forward are skipped;
- the outer graph does not emit an additional routed/shared add;
- the operator output is finalized directly.

The decision must be made outside the custom operation. If it were changed
only during eager replay, Dynamo could capture a graph that expects a shared
output while the real operator returns the already-fused result.

## Shared API v6 payload

The operator is loaded from the public package entry point:

```python
from cann_ops_transformer.ops import mega_moe_low_latency_with_shared
```

The integration builds a live `MegaMoeLowLatencySharedFusion` payload after
both shared linear layers finish loading. It contains:

- shared GMM1 and GMM2 weights;
- their E8M0 MX scales;
- an optional BF16 shared gate weight;
- a versioned descriptor identifying Shared API v6, Cube backend, and the
  FP4 or FP8 quantization format.

The payload is considered ready only when both weights, both scales, and the
descriptor are present. Until then, the fused shared path is not selected.

## Shared-expert quantization and layout

Shared API v6 requires the shared MLP to use the same MX family as the routed
experts. Checkpoints may nevertheless contain BF16 shared-expert weights when
their quantization rule includes `*mlp.experts*` but does not include the
standalone shared linear layers.

Weight preparation handles both cases:

1. If a shared linear already has `weight_scale`, its checkpoint MX weight and
   scale are reused.
2. If `weight_scale` is absent, the BF16 weight is quantized once after loading
   with `torch_npu.npu_dynamic_mx_quant` into MXFP4 or MXFP8.
3. GMM1 rows and scales are paired in the order required by Shared API v6.
4. Weights are exposed as explicit ND byte carriers representing the required
   FRACTAL_NZ physical storage. Scales are also forced to ND carriers.

The one-time conversion is outside the forward hot path.

### Current interleaving status

The routed-expert W1 preparation still calls
`interleave_mega_moe_low_latency_w1()` for both W4A4 MXFP4 and W8A8 MXFP8.
Shared GMM1 uses the Shared API v6 row-pairing rule. This PR does not remove
those transformations. Removing routed W1 interleaving requires confirming
the exact ABI of the installed ops-transformer build and updating the related
weight-preparation tests together.

## Operator selection

`AllGatherCommImpl` loads both public symbols when the feature is enabled. At
runtime it selects:

- `mega_moe_low_latency_with_shared` when the shared payload is ready;
- `mega_moe_low_latency` otherwise.

The shared arguments are appended only for the Shared API v6 call. The common
arguments retain the routed gate, correction bias, routed weights and scales,
expert offset, TopK configuration, scaling factor, output dtype, activation
clamp, and scoring activation.

## Enablement and fallback conditions

Global configuration checks require:

- `enable_mega_moe_low_latency=true`;
- Ascend 950PR or 950DT;
- expert parallelism;
- a compatible ops-transformer package;
- `enable_fused_mc2` disabled;
- MoE LoRA, EPLB, and redundant experts disabled.

Each forward also checks the layer and input contract, including:

- AllGather communication mode;
- W4A4 MXFP4 or W8A8 MXFP8 with group size 32;
- BF16 rank-2 hidden states and BF16 gate weights;
- hidden size from 1024 through 8192, aligned to 1024;
- expert count aligned to 32 and no more than 1024;
- one through 128 local experts;
- global softmax or sigmoid routing, TopK from 1 through 32, and
  renormalization enabled;
- no grouped/custom routing, dynamic EPLB, logical-to-physical expert map,
  or active profile dummy run;
- SiLU activation.

If a model has shared experts, the shared payload must be ready and the local
input must contain at most 16 tokens. A failed per-forward check falls back to
the regular MoE path rather than partially using the operator.

## Prefill and decode behavior

The feature does not unconditionally replace both phases.

- For a model without shared experts, prefill and decode may both select
  `MegaMoeLowLatency` when their input shapes satisfy the contract.
- For a model with shared experts, Shared API v6 currently requires at most 16
  local tokens. Decode normally satisfies this constraint, while ordinary
  prefill usually exceeds it and falls back to the regular AllGather MoE path.

The token-limit check is made before the hidden-state AllGather, so it applies
to the local token count seen by the rank.

## Source layout

| File | Responsibility |
| --- | --- |
| `ops/fused_moe/moe_utils.py` | Load public ops-transformer symbols and provide routed W1 layout conversion. |
| `ops/fused_moe/dataclass/fused_experts.py` | Define the Shared API v6 descriptor and live payload. |
| `ops/fused_moe/shared_experts.py` | Validate, quantize, lay out, and publish shared weights. |
| `ops/fused_moe/prepare_finalize.py` | Gather only hidden states for the routing-free path. |
| `ops/fused_moe/moe_comm_method.py` | Select the base/shared operator, bind arguments, and preserve collective milestones. |
| `ops/fused_moe/routed_experts.py` | Enforce the per-forward contract and select the low-latency path or fallback. |
| `ops/fused_moe/fused_moe.py` | Skip the external gate and standalone shared path when Shared API v6 is active. |

## Tests

The focused tests cover:

- loading the Shared API v6 symbol and reporting a missing symbol;
- exact base and shared operator argument binding;
- gathering hidden states exactly once without router information;
- skipping the external router and standalone shared path;
- Shared API v6 payload construction and readiness;
- the existing routed W1 and scale layout transformations.

Hardware performance validation should compare the base and fused-shared
paths with the same model, input/output lengths, concurrency, warm-up policy,
and profiler window. For decode analysis, verify the complete iteration rather
than summing only a selected operator region: host gaps, graph replay,
collective synchronization, and batch-shape differences can dominate the
end-to-end throughput result.
