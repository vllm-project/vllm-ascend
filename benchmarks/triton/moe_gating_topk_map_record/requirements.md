# MoE routing/map/record contract

## Target

- Entry: `moe_gating_topk_map_record` through `AscendFusedTopKRouter`.
- Current change: admit E=128 and EP-local physical ranges smaller than E.

## One-sentence semantics

From contiguous logits `[T,E]`, return mainline-equivalent routing weights
and mapped physical IDs `[T,K]`, and add one count per valid assignment
whose final physical ID lies in this rank's local expert range.

## ABI

| Item | Contract | Source | Status |
|---|---|---|---|
| Logits | Contiguous fp16/bf16/fp32 `[T,E]` | Current operator | Confirmed |
| Bias | Optional contiguous `[E]`, same dtype as logits | CANN TopK ABI | Confirmed |
| Routing table | Runtime int32 `[rows,E]`; lookup row `token % rows` | Mainline EPLB map | Confirmed |
| Expert load | int32 cumulative global physical-domain vector | Mainline EPLB record | Confirmed |
| Local range | `[local_expert_start, start + local_expert_count)`; count may differ from E | Mainline MoE count | Confirmed |
| Outputs | Weights in logits dtype; mapped IDs int32; load updated in place | Current operator | Confirmed |

## Function requirements

| ID | Requirement | Source | Status | Acceptance owner | Case |
|---|---|---|---|---|---|
| F1 | Physical IDs exactly match mainline map, including table replay | Mainline | Confirmed | Upstream | NPU graph replay |
| F2 | Local load increment exactly matches MoE-produced local counts | Mainline; user | Confirmed | Upstream/user | EP-local range NPU test, E=16/128 |
| F3 | Record-disabled and tail-padding assignments do not change load | User; mainline | Confirmed | User | NPU edge tests |
| F4 | Post-router ID rewrites and unsupported communication keep mainline | Mainline call chain | Confirmed | Upstream | Router fallback tests |

## Accuracy requirements

| ID | Requirement | Source | Status | Acceptance owner | Case |
|---|---|---|---|---|---|
| A1 | Weights match CANN TopK within `rtol=1e-4, atol=1e-5`; IDs/counts exact | Existing NPU tests | Confirmed | Upstream | E=16/128 NPU tests |

## Performance requirements

| ID | Requirement | Source | Status | Acceptance owner | Case |
|---|---|---|---|---|---|
| P1 | Existing E≤32 measurements remain documented; E=128 is not yet a measured speedup | User waived new performance run | Confirmed | User | `performance.md` |

Measurement boundary for prior results is summed `msprof op` component
Task Duration, not model latency. No performance acceptance threshold for
E=128 has been specified. The tile uses a fixed comparison budget divided by
the power-of-two local physical expert count; the model's E specializes the
kernel, but UB utilization alone cannot prove minimal latency.

## Validity and execution modes

Valid tokens must form a contiguous prefix. The fast path remains V2,
ALLGATHER, ungrouped renormalized softmax/sigmoid, without downstream ID
rewrites. Direct eager, graph replay and Router dispatch have separate tests.
A real TP2/EP2 model call is still required to confirm production dispatch;
synthetic operator tests do not substitute for it. Available vLLM 0.28/0.29
images are incompatible with the complete current branch, so the attempted
dummy-weight Qwen3-MoE smoke stopped during worker initialization.
