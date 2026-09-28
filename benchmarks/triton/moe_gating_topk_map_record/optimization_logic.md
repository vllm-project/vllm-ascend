# MoE TopK + map + record: comparison and optimization logic

## Scope and evidence rules

The four requested implementations are (1) upstream CANN TopK followed by
Triton EPLB map and record, (2) the private AscendC `moe_gating_top_k_with_map`
from [PR #3](https://github.com/maoxx241/vllm-ascend-rfc16468-private/pull/3),
(3) this branch's Triton fusion, and (4) the independent Triton fusion from
[PR #17574](https://github.com/vllm-project/vllm-ascend/pull/17574).
The comparison ranks only runs with the same device, software, input, semantics,
and measurement boundary. Published numbers from other environments are
context, not entries in that ranking.

The current isolated target is an otherwise idle Ascend 910B4-1, CANN 9.1.0,
PyTorch 2.10.0, torch-npu 2.10.0.post4, and Triton-Ascend 3.2.2. The pinned
upstream main baseline is `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`;
this branch is `triton/moe_gating_topk_map_record`. The exact environment and
compiler hash are in the experiment's `environment.json`/`performance.md`.
No second Triton version has been tested.

PR #3 cannot run on this target: its only tiling class returns `false` unless
the SoC is `ASCEND950` (A5); its own NPU tests are A5-gated. It accepts a
one-dimensional `[E]` map and fuses TopK + map, **not load recording**. Its
published A5 EP32/EP64 timings therefore cannot be divided by the present
910B4 data. The user has deferred A5 validation. This row remains `NOT RUN`,
not a loss or a win.

## Common data flow and contract

```text
logits[T,E] --softmax/sigmoid--> score --(optional bias for ranking)-->
stable logical TopK[T,K] --replica table[rows,E]--> physical IDs[T,K]
                  |                                  |
                  +--> selected score/renorm --> weights[T,K]
                                                     |
valid token count + recording enabled ----------------+--> local expert histogram
                                                            --> expert_load += counts
```

The common **910B4 comparison subset** is contiguous FP32 logits, E=8/16/32,
K=6/8, softmax/sigmoid, one group, renormalization enabled, no correction bias,
periodic routing table `[1024,E]`, valid tokens equal to T, and load recording
enabled. FP32 avoids comparing unlike output dtypes: PR #17574 returns FP32
weights while this branch returns the logits dtype. Exact physical IDs and
recorded counts, and weights within `rtol=1e-4, atol=1e-5`, gate performance.
Additional controls must cover bias, FP16/BF16, ties, duplicate physical IDs,
padding, recording disabled, T=65 tail, and graph table mutation where each
implementation supports those semantics. PR #3's 1-D mapping is comparable
only with a constant-across-rows table and a separate record kernel on A5.

The upstream production record kernel consumes local counts produced by the
downstream MoE operator. The standalone three-component profile gives it a
stable representative count vector. Summing its time with TopK and map is a
**component cost model**, not the contiguous production segment or model ITL.
That limitation applies equally whenever the native TopK+map path is augmented
with record to match the full functional boundary.

## Four implementations and expected costs

| Path | Kernel launches at the compared logical boundary | Distinctive work | Current status |
|---|---:|---|---|
| Upstream CANN + Triton map + Triton record | 3 | CANN TopK is fast, but writes logical IDs and maps/records separately | Same-host component profiles and three repeat controls complete |
| PR #3 AscendC TopK + map, plus record if full boundary is required | 2 | Native A5-only tiling; 1-D mapping, no built-in record | NOT RUN on 910B4 by design |
| This branch, PR #17579 | 1 for eligible small E/T | Stable ungrouped TopK, table lookup, program-local histogram, masked nonzero atomic per local physical expert | Accuracy and 910B4 msprof baseline in `performance.md` |
| PR #17574 | 1 on its eligible route | General grouped selection; hit matrix atomics over token-by-local-expert lanes | Same-host nine-case profile and three repeat controls complete |

The number of launches is only a cost indicator. The authoritative primary
number is `msprof op` **Task Duration** for each actual target kernel. Per
the user's hardware-team clarification, this duration already includes
the NPU task head/startup overhead; no separate head is added to it.
For a multi-kernel path, report both every component and their sum; do not
call the sum a measured end-to-end latency. `msprof op` performs its own
warmup, so each case/target is invoked once by the case script. Exclude
initial compilation and input creation. Preserve profiler summary, target
name, block count, and relevant Scalar/Vector/MTE metrics when available.
Small differences require additional balanced captures to establish
repeatability. A missing/NA duration is `NOT MEASURED`, never zero.

NPU stream events around the Python wrapper were used only for screening:
five warmups, then event before/after the wrapper, synchronize, five samples
(ten for selected spot checks), median. Multiple host enqueues and output
allocations lie inside that event interval, so this is **not** pure kernel
time or a measurement of the task head. The large
`stream-event - sum(Task Duration)` residual includes host enqueue/wait gaps,
not another task-head term. Host dispatch timing is also outside the user's
chosen primary metric. Isolating the head *within* Task Duration would need
a controlled minimal-task comparison, but is unnecessary for the present
ranking.

## Optimization hypotheses and decisions

1. **Decode is launch/intermediate dominated.** At T/E/K=64/16/8, the
   earlier 910B4 `msprof op` capture gave CANN TopK 6.16 µs, map 27.60 µs,
   record 1.76 µs (sum 35.52 µs), versus this branch's fused kernel 25.70
   µs. Only 9.82 µs of the much larger wrapper-level difference is device
   kernel-time saving. This supports fusion, but does not establish a pure
   launch latency.
2. **Fuse the producer-consumer chain, not every route.** The ungrouped
   kernel keeps per-token scores and selected logical IDs on chip, maps each
   selected ID immediately, and avoids the global logical-ID intermediate.
   It computes physical-expert hits locally across `BLOCK_T` tokens and
   reduces them before issuing at most one nonzero atomic per local physical
   expert per program. This differs materially from PR #17574's more general
   grouped-selection chain and token-by-local-expert atomic hit matrix. Any
   advantage from that difference is a hypothesis until the same-case
   profiler confirms it.
3. **Large T changes the bottleneck.** The earlier T/E/K=65536/16/8 capture
   gave a 2533.66 µs baseline component sum versus 2500.64 µs fused. The
   fused kernel schedules 2048 small programs; atomics disable backend
   AutoBlockify, and its per-program Scalar time (median 36.54 µs) exceeds
   Vector time (13.81 µs). Saved launches cannot compensate in the wrapper
   screen. Hence this branch dispatches T>4096 to the original route. PR
   #17574 uses a different T<=512 production guard measured on 910C; neither
   threshold should be universalized across E, SoC, or software versions.
4. **Avoid false comparisons.** PR #3's A5-only TopK+map result and PR
   #17574's published 910C/E256 grouped result use different hardware,
   domains and boundaries. They are design clues, not four-way speedups.

## Test matrix and result ledger

The full business matrix is T=64/128/256/512/64K/128K/256K/512K,
E=8/16/32, K=6/8, two scoring modes: 96 cases. Boundary controls include
T=65, 1024/2048/4096/8192/16384/32768. Do correctness across the common
domain before profiling. For profiler economy, the first matched-device
captures cover decode (T=64/128/256/512), the dispatch edge (4096),
and prefill (65536), varying E at representative K/scoring values. Each
reported row must identify source SHA and the exact profiler artifact;
unprofiled matrix cells remain explicitly missing.

T=256K and 512K are user-supplied **prefill business test sizes**, but that
does not prove that a particular serving configuration passes that many
tokens in one local MoE-router call: chunked prefill and parallel partitioning
must be traced separately. The direct PR #17574 kernel launch fails at
T=262144 because its grid/coreDim reaches 65536 (>65535), after 72 earlier
matrix cases passed. Its production dispatcher already falls back at T>512;
the direct-launch failure is a raw-kernel domain limit, not a demonstrated
serving-path crash.

| Comparison | Function/accuracy | Same-910B4 `msprof op` | Interpretation |
|---|---|---|---|
| Upstream vs PR #17579 | 96/96 business accuracy in prior run | Decode and prefill component profiles in `performance.md` | Eligible small-T win; large-T falls back |
| Upstream vs PR #17574 | 48/48 eligible decode cases passed; three FP32 edge controls passed | Nine-case matched-device screen, with three balanced controls | Faster at sampled T=64; slower in sampled T=256/512 |
| PR #17579 vs PR #17574 | Common FP32 case outputs/counts matched the baseline | Nine-case screen, with three balanced controls | #17579 faster in most sampled decode cases; #17574 faster at T=512/E=32/K=8 |
| PR #3 vs any 910B4 path | NOT RUN: ASCEND950-only | NOT RUN | Requires A5 and ABI-matched record boundary |

The detailed device-time table and artifact paths are in `performance.md`.
Three repeated `PipeUtilization` controls illustrate the crossover:

| T/E/K | Baseline component sum µs | This branch µs | PR #17574 µs | Interpretation |
|---|---:|---:|---:|---|
| 64/8/6 | 34.40 | 15.75 | 25.13 | Fusion wins; both use 32 programs, with lower per-program Scalar/Vector time on this branch |
| 256/16/8 | 39.38 | 56.94 | 64.54 | Both fusions lose device Task Duration despite fewer tasks |
| 512/32/8 | 48.78 | 94.29 | 83.05 | Both lose; PR #17574 wins over this branch with half as many programs (128 vs 256) |

At T=512/E=32, PR #17574 has **higher** per-program median Scalar and
Vector active times (9.35/5.87 µs versus this branch's 5.08/4.65 µs), yet
its total Task Duration is lower. This fits a grid/program-wave bottleneck:
halving the number of programs outweighs extra work within each program.
At T=256/E=16 both launch 128 programs, and this branch has lower per-program
Scalar/Vector times and lower total duration. These are evidence-backed
associations, not a claim that individual active-unit times add to total
duration. The earlier T≤4096 guard came from eager wrapper event screens;
if `msprof op` Task Duration is the acceptance metric, a narrower guard
should be evaluated with a broader same-metric scan rather than presumed
from those event results.
