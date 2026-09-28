# Operator requirements: MoE gating TopK + map + record

## Target

- Public route: `AscendFusedTopKRouter` with EPLB-enabled routed experts.
- Implementations: upstream CANN TopK + Triton map/record; private PR #3
  AscendC TopK+map; this branch PR #17579; independent Triton PR #17574.
- Task: function-first, same-boundary performance comparison and an
  optimization-logic document.

## One-sentence semantics

Given router logits, optional correction bias, an EPLB routing table and
validity/recording state, select stable TopK logical experts, produce
renormalized weights and mapped physical IDs, and increment local
physical-expert load counts for valid recorded tokens only.

## ABI

| Item | Contract | Source | Status |
|---|---|---|---|
| Inputs | Contiguous logits `[T,E]`, optional bias `[E]`, periodic table `[R,E]` int32, device flag/valid count, mutable load view | Current call sites | Observed/Confirmed for PR #17579 |
| Outputs | Weights `[T,K]`, physical IDs `[T,K]` int32, load increment | Current source/tests | Observed/Confirmed |
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

## Accuracy requirements

| ID | Requirement | Source | Status | Acceptance owner | Cases |
|---|---|---|---|---|---|
| ACC-001 | IDs/counts exact; weights `rtol=1e-4, atol=1e-5` on FP32 shared subset | Existing case generator | Observed/Confirmed | User | Business matrix |
| ACC-002 | Bias, FP16/BF16 and differing output dtype checked separately | PR ABIs | Inferred/Provisional | User | Edge suite |

## Performance requirements

| ID | Requirement | Source | Status | Acceptance owner | Cases |
|---|---|---|---|---|---|
| PERF-001 | Same-device, same-input `msprof op` device time; list components and sum for multiple kernels | User/profiling rule | Observed/Confirmed | User | Decode, cutoff, prefill |
| PERF-002 | Report regressions, unsupported and missing cells; do not rank historical A5/910C numbers against 910B4 | User/PR evidence | Observed/Confirmed | User | Full ledger |
| PERF-003 | Four-way quantitative acceptance threshold | Not supplied | Unknown | User | Pending |

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

## Inferred / provisional

Small T should benefit from removed launches and logical-ID intermediate.
Program-local count reduction may issue fewer atomics than #17574's
token-by-local-expert hit matrix; matched `msprof op` must test that claim.

## Unknown / acceptance needed

No A5 isolation environment, no cross-Triton-version A/B, no isolated pure
per-launch overhead, and no same-hardware four-way ranking including PR #3.
