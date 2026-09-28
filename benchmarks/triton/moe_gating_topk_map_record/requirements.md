# Operator requirements: MoE gating TopK + map + record

## Target

- Public route: `AscendFusedTopKRouter` with EPLB-enabled routed experts.
- Implementations: upstream CANN TopK + Triton map/record; private PR #3
  AscendC TopK+map; this branch PR #17579; independent Triton PR #17574.
- Task: replace the fused kernel's global atomics with one grid-owned record
  per program and a single-program reduction kernel, without changing TopK or
  expanding grouped-routing coverage. The mainline CANN + EPLB map + MoE
  count + EPLB record path is the correctness gold reference; PR #17574 is
  only a design/performance comparator.

## One-sentence semantics

Given router logits, optional correction bias, an EPLB routing table and
validity/recording state, select stable TopK logical experts, produce
renormalized weights and mapped physical IDs, and increment local
physical-expert load counts for valid recorded tokens only.

## ABI

| Item | Contract | Source | Status |
|---|---|---|---|
| Inputs | Contiguous logits `[T,E]`, optional bias `[E]`, periodic table `[R,E]` int32, device flag/valid count, mutable load view | Current call sites | Observed/Confirmed for PR #17579 |
| Outputs | Weights `[T,K]`, physical IDs `[T,K]` int32, load increment; internal grid-owned records `[P,E_record]` | User/current source | Observed/Confirmed; `E_record` is EP-local physical count on eligible paths |
| dtype | FP32 common comparison; #17579 weights follow logits dtype, #17574 returns FP32 | PR sources | Observed/Confirmed |
| layout/stride | Contiguous inputs on #17579 fast path | Wrapper guard | Observed/Confirmed |
| metadata | E=8/16/32, K=6/8, group=1, renorm, softmax/sigmoid; table can change across graph replay | User/tests | Observed/Confirmed for shared subset |
| out/workspace | Weights/IDs allocated per call, caller-owned load; PR #3 native workspace and 1-D map | PR sources | Observed/Confirmed |

## Function requirements

| ID | Requirement | Source | Status | Acceptance owner | Cases |
|---|---|---|---|---|---|
| FUNC-001 | Physical IDs and valid-token load counts exact | User/call sites | Observed/Confirmed | User/upstream | Business, duplicate map, padding, record off |
| FUNC-002 | Stable ties and safe masked tail | Tests/review | Observed/Confirmed | Upstream | Ties, all zero, T=65 |
| FUNC-003 | Graph replay reads updated table | Integration tests | Observed/Confirmed for #17579 | Upstream | Table mutation |
| FUNC-004 | Unsupported shape/layout uses original route | Production guards | Observed/Confirmed | Upstream | T>4096, E>32, noncontiguous |
| FUNC-005 | PR #3 needs separate record and row-invariant map for full-boundary comparison | PR #3 ABI | Inferred/Provisional | User | A5-only matched case missing |
| FUNC-006 | Kernel 1 writes exactly one grid-owned record per program with ordinary stores and no global atomic; kernel 2 alone accumulates the final load without atomic | User | Observed/Confirmed | User | Source/IR scan, NPU execution, repeated loads |
| FUNC-007 | The grid record indexes the same local physical-expert domain as the MoE-returned `expert_tokens`, not logical expert IDs | Mainline `AscendEplbLayerState` and `_record_v2_eplb_load` | Observed/Confirmed | Upstream/user | Redundant map, local range, actual MoE count |
| FUNC-008 | Fast path remains ungrouped, renormalized and excludes ID-rewrite, padding or communication paths whose downstream expert count is not the router-local histogram | User/mainline guards | Observed/Confirmed for existing guards; All2All mismatch inferred | User/upstream | Router guard, AllGather and All2All controls |
| FUNC-009 | Dynamic routing-table updates and graph replay affect both mapped IDs and recorded load without a host scalar read | User/current tests | Observed/Confirmed | User | Graph replay with record on/off and changed table |

## Accuracy requirements

| ID | Requirement | Source | Status | Acceptance owner | Cases |
|---|---|---|---|---|---|
| ACC-001 | IDs/counts exact; weights `rtol=1e-4, atol=1e-5` on FP32 shared subset | Existing case generator | Observed/Confirmed | User | Business matrix |
| ACC-002 | Bias, FP16/BF16 and differing output dtype checked separately | PR ABIs | Inferred/Provisional | User | Edge suite |
| ACC-003 | Grid-record reduction, per-step load and cumulative expert load exactly equal the mainline MoE operator's returned count after `group_list_type` conversion | User | Observed/Confirmed | User | Actual `npu_moe_init_routing` count, nonzero initial load, record on/off |

## Performance requirements

| ID | Requirement | Source | Status | Acceptance owner | Cases |
|---|---|---|---|---|---|
| PERF-001 | Same-device, same-input `msprof op` device time; list components and sum for multiple kernels | User/profiling rule | Observed/Confirmed | User | Decode, cutoff, prefill |
| PERF-002 | Report regressions, unsupported and missing cells; do not rank historical A5/910C numbers against 910B4 | User/PR evidence | Observed/Confirmed | User | Full ledger |
| PERF-003 | Four-way quantitative acceptance threshold | Not supplied | Unknown | User | Pending |
| PERF-004 | Compare one-kernel atomic baseline with two-kernel no-atomic candidate by per-kernel `msprof op` Task Duration and their sum across tiny/decode/prefill shapes | User | Observed/Confirmed | User | T=1/2/4/8/16/32, 64/128/256/512, prefill, E=8/16/32 |

### Performance acceptance scope

- Boundary: each actual kernel's `msprof op` Task Duration, including its NPU
  task head/startup cost per the user's hardware-team clarification. A three-kernel
  sum is a component cost model, not measured contiguous E2E: production
  record follows the downstream MoE operator.
- Metric: baseline component sum / candidate device duration; keep raw µs.
- Business matrix: T=64/128/256/512/64K/128K/256K/512K;
  E=8/16/32; K=6/8; softmax/sigmoid (96 cases), supplied by the user.
- Scope: every profiled case is reported; no aggregate weights or universal
  percentage were supplied. Missing profiles remain `NOT MEASURED`.
- Gradient controls: T=65, 1024/2048/4096/8192/16384/32768.
- Time-box: at most seven kernel-optimization rounds from the earlier task;
  this comparison is evidence work, not a new tuning round.

## Validity / boundary / sentinel

`0 <= valid_tokens <= T`; padded rows do not record; recording disabled
leaves load unchanged; duplicate physical IDs count repeated routes; lowest
logical index wins ties. #17579 fast path requires E<=32, K<=8, T<=4096.
PR #17574 production route requires T<=512; its direct kernel exceeds the
910B4 max coreDim at T=262144 and must not be substituted for its fallback.
For the new two-kernel path, `E_record` is the EP-local physical-expert count,
which can differ from logits' logical `E` under redundant EPLB. Per user
direction, version 1 enables only configurations in which the requested
`[P,E]` grid record has exactly the mainline record domain; other cases
fall back. All2All downstream counts include assignments received from
other ranks, so a router-local histogram cannot replace them without a
distributed reduction; exclude that communication mode pending proof.

## Execution modes

The matched-device experiment is eager standalone in an isolated 910B4
container. #17579 graph replay is checked separately. Private PR #3's only
tiling class requires ASCEND950 (A5), so it is deferred by user direction.

## Observed

#17579 passed its prior 96/96 business accuracy cases and 18 NPU integration
tests; earlier same-910B4 profiles are in `performance.md`. The current
#17574 direct-kernel sweep passed 72 cases through T=131072, including all
48 production-eligible decode cases; T=262144 failed at grid/coreDim 65536.
Three additional supported FP32 controls passed: T=65 masked tail with
duplicate mapping, T=128 ties/bias/padding, and recording disabled. The
existing edge-suite case with host-integer valid-token count is outside
#17574's ABI, which explicitly requires a device scalar.
The standalone **timing** baseline supplies a stable representative
`expert_tokens` vector to isolate the record kernel's cost. Accuracy now uses
the actual `npu_moe_init_routing_v2` count-mode output and applies the existing
EPLB record kernel to nonzero initial load. Synthetic partial-prefix controls
are labelled separately because they do not emulate the full MoE dispatcher.

## Inferred / provisional

Small T should benefit from removed launches and logical-ID intermediate.
Program-local count reduction may issue fewer atomics than #17574's
token-by-local-expert hit matrix; matched `msprof op` must test that claim.

## Unknown / acceptance needed

No A5 isolation environment, no cross-Triton-version A/B, no isolated pure
per-launch overhead, and no same-hardware four-way ranking including PR #3.
