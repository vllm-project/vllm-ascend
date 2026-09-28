# MoE gating TopK + EPLB mapping/recording performance

## Current source status — CANN TopK + two Triton kernels

The current branch now uses CANN `moe_gating_top_k` for scoring/TopK, followed
by a grid-owned Triton map/record kernel and a single-writer Triton reduction.
Neither Triton kernel uses global atomics. The isolated Ascend 910B4 target
passed all 96 business-shape accuracy cases, 14 smoke cases, six edge cases,
routing-table graph replay, and 24/24 NPU tests (including the large-prefill
router dispatch) with this exact source. The historical rounds
below describe earlier implementations and **are not performance claims for
the current source**. Exact-source `msprof op` timings will be added in a
follow-up after the call chain is committed.

Environment: Ascend 910B4-1, CANN 9.1.0, PyTorch 2.10.0, torch-npu
2.10.0.post4, Triton-Ascend 3.2.2, one otherwise idle NPU. Upstream main
base: `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`. The isolated test
container uses the same CANN gating source as this base and a copy of
its EPLB Triton kernels. All files and logs are under the task directory on
the validation host; no shared runtime was modified.

Timing boundaries:

- The matrix below uses NPU stream events around the Python wrapper and
  includes gaps while the host enqueues kernels. It is a useful eager routing
  segment screen, not pure kernel execution time. Five post-warmup samples per
  implementation and case; entries are the median across K=6/8 and
  softmax/sigmoid at each T/E.
- `msprof op` measures each target kernel's device time, with its own warmup.
  A three-kernel sum is a cost model because production load recording follows
  downstream MoE execution. `visualize_data.bin` is not needed.
- The latest main record kernel consumes local counts produced by the MoE
  operator. The standalone benchmark supplies a stable representative count
  vector so that only its recording kernel is included in this segment.

## Round 1 — one kernel, program-local physical-expert histogram

One Triton program computes scores, stable TopK, replica-table mapping and
local-expert histogram, then issues at most one atomic add per local physical
expert. This is an independent implementation; PR #17574 is used only as a
design comparison. All 96 business cases passed against the fresh-main CANN
TopK + mapping path: physical IDs and loads exact, weights within 1e-4/1e-5.
Five additional focused cases passed: tie and all-zero scores, repeated
physical IDs, padding exclusion, recording disabled, device-scalar versus
host-integer valid-token count, correction bias, fp16, and bf16.
The integrated NPU test suite passed 13/13 cases, including routing-table
updates across graph replays.

| T | E | Main 3-kernel stream µs | One kernel stream µs | Main / one |
|---:|---:|---:|---:|---:|
| 64 | 8 | 510.4 | 346.2 | 1.47× |
| 64 | 16 | 518.4 | 387.0 | 1.34× |
| 64 | 32 | 514.5 | 361.5 | 1.42× |
| 128 | 8 | 614.9 | 370.9 | 1.66× |
| 128 | 16 | 605.6 | 363.7 | 1.67× |
| 128 | 32 | 507.5 | 312.9 | 1.62× |
| 256 | 8 | 508.9 | 312.5 | 1.63× |
| 256 | 16 | 510.0 | 313.6 | 1.63× |
| 256 | 32 | 515.1 | 310.5 | 1.66× |
| 512 | 8 | 503.1 | 322.7 | 1.56× |
| 512 | 16 | 593.8 | 361.5 | 1.64× |
| 512 | 32 | 587.3 | 356.2 | 1.65× |
| 65536 | 8 | 2267.2 | 2497.3 | 0.91× |
| 65536 | 16 | 2279.1 | 2467.6 | 0.92× |
| 65536 | 32 | 2410.4 | 2636.7 | 0.91× |
| 131072 | 8 | 4334.2 | 4761.0 | 0.91× |
| 131072 | 16 | 4390.8 | 4679.5 | 0.94× |
| 131072 | 32 | 4558.5 | 5036.5 | 0.91× |
| 262144 | 8 | 8534.5 | 9216.8 | 0.93× |
| 262144 | 16 | 8675.8 | 9052.0 | 0.96× |
| 262144 | 32 | 8923.2 | 9746.9 | 0.92× |
| 524288 | 8 | 16959.0 | 18123.9 | 0.94× |
| 524288 | 16 | 17246.5 | 17821.1 | 0.97× |
| 524288 | 32 | 17825.4 | 19258.0 | 0.93× |

The small-T improvement is consistent with two fewer launches and no
intermediate logical-ID tensor. For large T, the msprof component evidence
below identifies per-program scalar work and reduced block aggregation as
the main costs; the saved launches no longer compensate for them.

At T=64, E=16, K=8, softmax, no bias, recording on, msprof reports:

| Kernel | Device time µs | Block dim |
|---|---:|---:|
| Main CANN `MoeGatingTopK` | 6.16 | 32 |
| Main Triton map | 27.60 | 2 |
| Main Triton record | 1.76 | 1 |
| Round-1 fused Triton | 25.70 | 32 |

The summed main kernels take 35.52 µs versus 25.70 µs fused. The same
case's stream-event medians are 510.54 and 303.52 µs, and host-dispatch
medians are 436.16 and 245.97 µs. Thus a large part of the eager gain is
enqueue/scheduling overhead, not faster TopK arithmetic; the CANN TopK
kernel itself is much faster than the fused kernel's full computation.

At T=65536, E=16, K=8, softmax, the device-time breakdown changes:

| Kernel | Device time µs | Block dim |
|---|---:|---:|
| Main CANN `MoeGatingTopK` | 1228.32 | 40 |
| Main Triton map | 1303.88 | 40 |
| Main Triton record | 1.46 | 1 |
| Round-1 fused Triton | 2500.64 | 2048 |

The prefill kernel sum is effectively at parity, while the stream screen
favors main. The atomics in the fused kernel disable Triton-Ascend
AutoBlockify; 2048 small programs are scheduled instead of the baseline
map's 40 blocks. `PipeUtilization.csv` shows a median 36.54 µs scalar time
versus 13.81 µs vector time per fused program. This supports keeping the
main path for large batches. A CANN TopK + atomic map/record two-kernel
route is therefore not assumed faster merely because it saves a launch.

### Gradient cases for the dispatch threshold

Same setup and medians as above, across K=6/8 and both scoring modes:

| T | E | Main stream µs | One kernel stream µs | Main / one |
|---:|---:|---:|---:|---:|
| 1024 | 8 | 547.0 | 421.2 | 1.30× |
| 1024 | 16 | 553.7 | 368.6 | 1.50× |
| 1024 | 32 | 572.9 | 338.1 | 1.69× |
| 2048 | 8 | 544.8 | 345.5 | 1.58× |
| 2048 | 16 | 546.7 | 354.0 | 1.54× |
| 2048 | 32 | 528.2 | 350.2 | 1.51× |
| 4096 | 8 | 540.2 | 438.4 | 1.23× |
| 4096 | 16 | 544.7 | 423.3 | 1.29× |
| 4096 | 32 | 539.6 | 443.5 | 1.22× |
| 8192 | 8 | 543.1 | 561.7 | 0.97× |
| 8192 | 16 | 545.8 | 560.0 | 0.97× |
| 8192 | 32 | 542.5 | 590.5 | 0.92× |
| 16384 | 8 | 664.4 | 834.7 | 0.80× |
| 16384 | 16 | 676.5 | 822.0 | 0.82× |
| 16384 | 32 | 698.5 | 871.7 | 0.80× |
| 32768 | 8 | 1200.9 | 1417.0 | 0.85× |
| 32768 | 16 | 1237.5 | 1407.4 | 0.88× |
| 32768 | 32 | 1267.2 | 1488.7 | 0.85× |

All 36 cases at T≤4096 were faster with the fused kernel. Eight of twelve
cases at T=8192 were slower, so the model integration uses a T=4096 guard.
This is specific to E≤32 and this NPU/software environment, not a general
threshold for larger expert counts.

Final-source spot check after formatting (same device, ten post-warmup
repetitions, T/E/K=64/16/8): main 538.93 µs versus fused 371.62 µs (1.45×).
At T/E/K=65536/16/8, main 2612.04 µs versus fused 2749.08 µs (0.95×).
These runs preserve the small-T win and large-T fallback decision; the
absolute eager timings vary between runs and should not be read as device
kernel durations.

## Round 2 — preserve fallback for non-contiguous router inputs

The eligibility guard now sends non-contiguous logits or correction bias to
the existing operator path. The fused kernel and its contiguous-input launch
arguments are unchanged, so the Round-1 timing matrix still applies to the
eligible cases. The new fallback cases are checked independently in the NPU
test suite (14/14 passed); no throughput claim is made for those inputs.
The unchanged contiguous T/E/K=64/16/8 path was remeasured with ten
post-warmup stream-event samples: main 543.76 µs, fused 372.19 µs (1.46×),
consistent with Round 1.

## Round 3 — finite softmax state for masked tail rows

An out-of-bounds token row previously reduced an all-`-inf` logit vector,
creating an internal `-inf - (-inf)` NaN before masked output stores. The
softmax now uses a finite zero maximum for an empty row and a nonzero
denominator. The new T=65 case exercises the non-divisible tail; the NPU
suite passed 18/18 and the full 96-case accuracy matrix passed 96/96.

Five post-warmup stream-event samples per implementation and case, medians
across K=6/8 and softmax/sigmoid at each T/E (the ratio is the median of
per-case ratios):

| T | E | Main µs | Fused µs | Main / fused |
|---:|---:|---:|---:|---:|
| 64 | 8 | 513.4 | 375.8 | 1.38× |
| 64 | 16 | 525.1 | 353.0 | 1.53× |
| 64 | 32 | 512.5 | 400.7 | 1.30× |
| 128 | 8 | 506.8 | 334.4 | 1.52× |
| 128 | 16 | 507.4 | 330.0 | 1.54× |
| 128 | 32 | 496.3 | 325.3 | 1.52× |
| 256 | 8 | 496.3 | 328.7 | 1.52× |
| 256 | 16 | 506.2 | 322.6 | 1.57× |
| 256 | 32 | 507.4 | 328.3 | 1.55× |
| 512 | 8 | 504.0 | 335.2 | 1.51× |
| 512 | 16 | 519.4 | 342.1 | 1.52× |
| 512 | 32 | 515.4 | 336.2 | 1.54× |
| 65536 | 8 | 2272.9 | 2524.3 | 0.89× |
| 65536 | 16 | 2320.3 | 2498.4 | 0.92× |
| 65536 | 32 | 2373.8 | 2682.9 | 0.89× |
| 131072 | 8 | 4338.3 | 4755.2 | 0.91× |
| 131072 | 16 | 4416.0 | 4715.0 | 0.93× |
| 131072 | 32 | 4557.9 | 5059.6 | 0.90× |
| 262144 | 8 | 8533.5 | 9235.7 | 0.92× |
| 262144 | 16 | 8678.2 | 9133.3 | 0.94× |
| 262144 | 32 | 8942.5 | 9817.6 | 0.91× |
| 524288 | 8 | 16930.1 | 18122.1 | 0.93× |
| 524288 | 16 | 17230.0 | 17921.3 | 0.95× |
| 524288 | 32 | 17822.5 | 19361.8 | 0.92× |

All 48 decode business cases still favor fusion. A 72-case gradient scan
found all 36 cases at T≤4096 faster; T=8192 had three regressions, so the
dispatch guard stays at 4096. `msprof op` measured the repaired
T/E/K=64/16/8 fused device kernel at 26.10 µs, block dim 32, versus
25.70 µs in Round 1. This single 0.40 µs difference is too small to
attribute to the repair without repeated profiles. The tested compiler was
`bishengir-compile` 1.2.0, SHA256
`89655a56941efe9a184e4d5dfccb783ad88458707827ff6fdd146e7bd2d4af5c`.

## Round 4 — CI typing correction in wrapper validation

The CI mypy check reported a fixed-length tuple inferred as four elements
and then extended to five. The device-validation tuple now has a fixed five
slots from construction; the Triton kernel and launch configuration are
unchanged. The final source passed the 18-case NPU suite. Ten post-warmup
stream-event samples confirm the eligible route remains faster:

| T | E | K | Main µs | Fused µs | Main / fused |
|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 8 | 550.11 | 380.93 | 1.44× |
| 4096 | 32 | 8 | 541.32 | 494.22 | 1.10× |

These are wrapper-level screens, not new device-kernel measurements. The
type-check result itself awaits the next CI run.

## Round 5 — align the CPU forward-flow test fixture

CI pre-commit, including mypy, passed after Round 4. The selected CPU UT
then found that its mocked MoE configuration lacked the `dp_size` and
`pcp_size` fields now used by the eligibility guard. Adding these two fields
to the fixture made all eight forward-flow parameterizations pass. The
kernel and production dispatch are unchanged; the 18-case NPU suite also
passed. A ten-sample wrapper screen at T/E/K=64/16/8 measured main
573.29 µs versus fused 381.43 µs (1.50×), consistent with prior rounds.

## Cross-implementation re-evaluation — no kernel tuning round

The user clarified that **`msprof op` Task Duration is the primary performance
metric and includes the NPU task's head/startup overhead**. The earlier
stream-event and host-dispatch screens above remain historical wrapper-level
observations, but do not rank device task performance and do not estimate a
separate per-task head cost. The `stream-event - task-duration` residual
includes host enqueue gaps and must not be called the task head.

The new same-host comparison used an isolated validation container on
Ascend 910B4-1, CANN 9.1.0, Triton-Ascend 3.2.2, and `bishengir-compile`
1.2.0 (SHA256 above). The branch source was `b7c2fb716` and its copied kernel
SHA256 was `5d5ae2b9aceac315c7e716832168c4352bd8575d82fdbf061d374a715a863ab7`.
The independent PR #17574 head was `3bdd2d7577d6d8d45ac79fd513f19490a0fa7d2b`;
the exact extracted Triton file SHA256 was
`0b19253667f411b9529877e56a7356244bdbfd30529f019e32c8a1301a4121fd`.
All compared inputs were FP32, one group, renormalized, no bias, recording on,
with identical seeded logits and periodic EPLB table. Each `msprof op`
invocation ran the case script once and used profiler-owned warmup. Cases
were parallelized over physical NPU 0–4 with
`ASCEND_RT_VISIBLE_DEVICES=<physical-id>`; **all implementations of one case
ran serially on the same card**. Profiles have separate output directories
and Triton caches. `npu-smi` confirmed the processes on the requested cards.
The primary profiles and raw logs remain in the isolated validation task
directory; the manifest is
`msprof_results.jsonl` plus `msprof_results_d*.jsonl`. The control captures
with `--aic-metrics=PipeUtilization` are in `msprof_controls_d*.jsonl`, with
per-core summaries in `pipe_components.jsonl`.

The baseline is CANN TopK + Triton map + Triton record. Its component sum is
a cost model: record follows downstream MoE in production. `main / candidate`
below is a device-task-cost ratio, **not** contiguous serving latency. One
default `msprof op` capture was made per target/case in this broad screen;
the three decision cases below also have balanced repeat controls.

| T | E | K | Scoring | Physical NPU | CANN TopK µs | Map µs | Record µs | Main sum µs | Our #17579 µs | #17574 µs | Main/ours | Main/#17574 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8 | 6 | softmax | 0 | 6.40 | 26.48 | 1.82 | 34.70 | 16.64 | 24.68 | 2.09× | 1.41× |
| 64 | 16 | 8 | sigmoid | 0 | 5.46 | 27.38 | 1.72 | 34.56 | 24.52 | 28.50 | 1.41× | 1.21× |
| 128 | 32 | 8 | softmax | 0 | 7.96 | 30.18 | 1.74 | 39.88 | 38.74 | 44.18 | 1.03× | 0.90× |
| 256 | 8 | 8 | sigmoid | 0 | 8.68 | 25.52 | 1.82 | 36.02 | 55.68 | 65.88 | 0.65× | 0.55× |
| 256 | 16 | 8 | softmax | 0 | 10.60 | 27.42 | 1.76 | 39.78 | 58.42 | 65.26 | 0.68× | 0.61× |
| 512 | 16 | 6 | sigmoid | 1 | 13.28 | 28.98 | 1.80 | 44.06 | 57.22 | 69.00 | 0.77× | 0.64× |
| 512 | 32 | 8 | softmax | 2 | 16.02 | 31.54 | 1.80 | 49.36 | 95.66 | 84.68 | 0.52× | 0.58× |
| 4096 | 32 | 8 | softmax | 3 | 82.76 | 116.92 | 1.88 | 201.56 | 214.42 | fallback | 0.94× | — |
| 65536 | 16 | 8 | softmax | 4 | 1230.94 | 1304.52 | 1.58 | 2537.04 | fallback | fallback | — | — |

`fallback` means the PR's production dispatch retains the baseline, not that
its raw kernel was assigned zero duration. #17579's guard is T≤4096;
PR #17574's is T≤512. PR #17574's raw kernel passed 72 of the 96 direct
business accuracy cases through T=131072, including all 48 eligible decode
cases. Direct T=262144 reached grid/coreDim 65536 and failed the device's
65535 maximum. Production T>512 falls back, so this is a raw-kernel limit,
not a demonstrated production failure. Whether T=256K occurs as one local
MoE call under chunking/parallelism is not yet established.
Three additional supported FP32 controls passed: T=65 masked tail with
duplicate mapping, T=128 ties with bias and padding, and recording disabled.
The original edge-suite host-integer valid-token case is outside #17574's
device-scalar ABI; it was not counted as a numerical mismatch.

Three representative cases were recaptured with balanced candidate order
and `PipeUtilization`; all reported `Current Freq = Rated Freq = 1650 MHz`
and the requested physical Device Id. The baseline components were captured
once per case in this control mode; the candidate numbers are two-capture
medians, so a small margin still needs more samples for a stability claim.

| T/E/K, scoring | Card | Baseline component sum µs | Ours µs (two runs) | #17574 µs (two runs) | Main/ours | Main/#17574 |
|---|---:|---:|---:|---:|---:|---:|
| 64/8/6, softmax | 1 | 34.40 | 15.75 (15.76, 15.74) | 25.13 (24.90, 25.36) | 2.18× | 1.37× |
| 256/16/8, softmax | 0 | 39.38 | 56.94 (56.32, 57.56) | 64.54 (63.98, 65.10) | 0.69× | 0.61× |
| 512/32/8, softmax | 2 | 48.78 | 94.29 (95.42, 93.16) | 83.05 (83.00, 83.10) | 0.52× | 0.59× |

The T=512/E=32 reversal between the two fused kernels is informative:
our kernel launches 256 programs versus #17574's 128. In this control,
our per-program median AIV Scalar/Vector times were 5.08/4.65 µs, versus
PR #17574's 9.35/5.87 µs. Thus PR #17574 does *more* work per program but schedules
half as many programs; its lower total Task Duration is consistent with
program-count/wave overhead dominating. At T=256/E=16, both launch 128
programs and our lower median Scalar/Vector times (4.92/4.52 µs versus
5.39/5.22 µs) align with its lower Task Duration. These are supported
associations, not proof that one instruction alone causes the difference;
MTE/Scalar/Vector active times can overlap and must not be added mechanically.

The task-level result changes the optimization decision: this branch's
T≤4096 guard was chosen from eager wrapper event screens, whereas `msprof op`
shows device-cost regressions already at sampled T=256/512 and near parity
at T=128. A narrower guard may be appropriate **if device Task Duration is
the acceptance metric**, but changing production dispatch requires an
explicit decision about eager versus graph/serving latency and a broader
same-metric shape scan. No kernel or dispatch change was made in this
comparison-only pass.

Private PR #3 is not in this table: its AscendC tiling accepts only
`ASCEND950` (A5), while this machine is 910B4, and it fuses only TopK + a
one-dimensional map, not record. The user asked to defer A5 validation.

## Round 6 — grid-owned record, two kernels, no atomic

The new kernel 1 partitions contiguous tokens among programs, carries one
`[E_record]` grid-owned count across all its token tiles, and stores one
non-overlapping row. Kernel 2 reduces those rows and adds them to cumulative
load in one program. This round retains the original stable TopK and uses
`BLOCK_T=2` for T≤64, `8` for 65–512, and `32` above 512, with at most the
40 vector-core-count programs for non-tiny cases. No grouped-route expansion.

Correctness gold was strengthened from a synthetic histogram to CANN TopK +
mainline EPLB mapping + actual `torch_npu.npu_moe_init_routing_v2` count-mode
output + the existing EPLB record kernel. The isolated 910B4 container passed
22 NPU tests, including nonzero initial load, EP-local physical offset,
record-off, All2All/domain fallback and graph replay after device-side table,
flag and valid-count changes. The standalone smoke set passed 14/14, the
edge set 6/6, and T=65536/E=16 plus T=524288/E=32 actual-MoE-count controls
passed. The T=524288 run verifies function, not production routing eligibility.
`TRITON_KERNEL_DUMP=1`, `TRITON_DEBUG=1` and forced recompilation produced
TTIR and NPU IR for both kernels; neither contains an atomic operation.

The following are same-case, same-physical-card `msprof op` Task Durations
with profiler-owned warmup and `PipeUtilization`. Each implementation of a
case ran serially on its assigned card. The mainline and new two-kernel
columns sum component Task Durations; they are a device-task cost model, not
contiguous serving latency. Mainline record timing uses a stable representative
count vector, while **correctness** above uses the actual MoE-returned vector.
The old atomic version is this branch's pre-redesign source, not PR #17574.

| T/E/K, scoring | Card | Mainline: CANN + map + record µs | Old atomic µs | Grid routing µs | Grid reduce µs | New sum µs | Main/new |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64/16/8, softmax | 0 | 5.76 + 27.90 + 1.68 = 35.34 | 24.20 | 25.68 | 2.24 | 27.92 | 1.27× |
| 128/32/6, softmax | 0 | 7.90 + 32.76 + 1.74 = 42.40 | 30.04 | 33.88 | 2.08 | 35.96 | 1.18× |
| 256/8/6, sigmoid | 1 | 8.70 + 26.40 + 1.72 = 36.82 | 36.00 | 35.12 | 2.20 | 37.32 | 0.99× |
| 512/32/8, sigmoid | 1 | 13.86 + 31.26 + 1.68 = 46.80 | 91.44 | 61.96 | 2.10 | 64.06 | 0.73× |
| 4096/32/8, softmax | 3 | 82.16 + 116.62 + 1.58 = 200.36 | 216.78 | 212.06 | 2.12 | 214.18 | 0.94× |
| 65536/16/8, softmax | 3 | 1227.94 + 1306.62 + 1.66 = 2536.22 | 2538.00 | 2478.90 | 2.36 | 2481.26 | 1.02× |
| 262144/32/8, sigmoid | 4 | 3784.04 + 5776.12 + 1.96 = 9562.12 | unavailable | 10445.98 | 2.10 | 10448.08 | 0.92× |

At T=262144 the old atomic profile reached its target but its profiler
analysis did not finish after about ten minutes; that task-owned capture was
terminated and **no old-atomic duration is claimed**. The new and mainline
captures completed separately. All artifact logs are retained under
`/home/shy/moe-gating-topk-map-record/grid-record-v1/` in the isolated
`va-blue-b4-moe-gating-01` container's mounted task directory; the script is
`profile_msprof_op.sh`.

The T=512 component comparison explains the incomplete benefit. Old atomic
routing used 256 programs with median per-program AIV/Scalar/Vector times
11.32/5.08/4.46 µs. Grid-owned routing used 40 programs but had median
57.59/27.98/12.65 µs, plus 6.34 µs MTE3 active time. Those units overlap;
their times are not additive. Fewer programs and no atomic reduced the old
91.44 µs task to 61.96 µs, yet the long per-program token loop left it above
the 46.80 µs three-task mainline cost. This motivated the next round's
single-factor token-tile experiment, not a TopK algorithm change.

## Round 7 — bound one grid's routing work to fewer token tiles

Only the token-tile schedule changed; the two kernels, TopK, mapping, record
math, grid ownership, and public ABI did not. The selected rule takes the
next power of two of `ceil(T / num_grids)`, bounded to `[8,64]` for non-tiny
cases; T≤64 retains two tokens/grid. This generally lets a program process
its owned decode tokens in one tile without making the live tensor unbounded.
The vector-core count comes from Triton's device-property helper, not a
hard-coded 910B4 constant.

The one-factor `msprof op` screens narrowed the schedule:

| Case | Variant | Kernel-1 Task Duration µs | Interpretation |
|---|---|---:|---|
| T=8, E=16, K=8 | 2 / 4 / 8 tokens per grid | 18.48 / 23.90 / 28.58 | 2 wins among compilable variants |
| T=8, E=16, K=8 | 1 token per grid | unavailable | Triton-Ascend compiler assertion in `BlockPtrAnalysis::parseSelect`; not a timing result |
| T=512, E=32, K=8, sigmoid | Grid factor 1 / 2 / 4, tile 8 | 62.36 / 64.36 / 100.78 | more programs do not fix per-program work |
| T=512, E=32, K=8, sigmoid | tile 8 / 16 / 32, 40 grids | 62.36 / 44.32 / 57.64 | one 16-row tile avoids a second loop; 32 increases live work |
| T=65536, E=16, K=8 | Grid factor 1 / 2 / 4, tile 32 | 2482.70 / 2475.92 / 2479.10 | differences below 1% |
| T=65536, E=16, K=8 | tile 32 / 64, 40 grids | 2482.70 / 2305.16 | 64-row tile reduces loop overhead |
| T=262144, E=32, K=8 | tile 64, grid factor 1 / 2 / 4 | 9740.52 / 9818.98 / 9808.58 | factor 1 remains best |

The T=512 `PipeUtilization` control shows median per-program AIV/Scalar/
Vector times change from 57.59/27.98/12.65 µs at tile 8 to
40.27/21.32/9.09 µs at tile 16. The component captures are on the same
910B4 SKU but different physical cards; the independent same-card Task
Duration screen gives the stronger evidence for the schedule choice. Active
unit times overlap and must not be summed. Tile 32 raised median Scalar to
34.21 µs, consistent with excess live/masked work. No global atomic was
introduced: final-source TTIR and NPU IR both scan clean for the two kernels.

Final-source correctness in the isolated container: NPU UT **22/22**, smoke
**14/14**, edge **6/6**, graph replay passed, and the complete business
accuracy matrix **96/96** passed. Each all-valid case compared physical IDs,
weights and nonzero-initialized cumulative load with CANN TopK, EPLB mapping,
actual MoE-returned counts, and existing EPLB record. A copied current CPU
forward-flow test could not be collected against the older container checkout
because it lacks `vllm_ascend.ops.fused_moe.dataclass.shared_experts`; its
older in-image version lacks the current config fixture and fails before this
kernel path. Neither is claimed as a CPU test pass; upstream CI remains the
applicable integration check.

Same-card `msprof op` results follow. The old atomic and
PR #17574 columns are for context only; PR #17574 is **not** the correctness
gold. The PR #17574 direct Triton source was its pinned `3bdd2d7577d6d8d45ac79fd513f19490a0fa7d2b` version.

| T/E/K, scoring | Card | Mainline sum µs | Old atomic µs | New routing + reduce µs | New sum µs | PR #17574 µs | Main/new |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64/8/6, softmax | 5 | 35.26 | 16.62 | 20.18 + 2.40 | 22.58 | 25.34 | 1.56× |
| 512/32/8, sigmoid | 1 | 46.80 | 91.44 | 44.80 + 2.14 | 46.94 | 82.46 | 1.00× |
| 4096/32/8, softmax | 3 | 200.36 | 216.78 | 197.84 + 2.14 | 199.98 | fallback | 1.00× |
| 65536/16/8, softmax | 3 | 2536.22 | 2538.00 | 2309.60 + 2.14 | 2311.74 | fallback | 1.10× |
| 262144/32/8, sigmoid | 4 | 9562.12 | unavailable | 9726.78 + 2.06 | 9728.84 | fallback | 0.98× |
| 524288/32/8, sigmoid | 6 | 19054.96 | not run | 19315.36 + 2.06 | 19317.42 | fallback | 0.99× |

Unchanged-schedule T=128/E=32 and T=256/E=8 results remain in Round 6.
`fallback` for PR #17574 means its production route is T≤512. The new
branch's production guard remains T≤4096: all 64K–512K entries here are raw
operator measurements, **not** claimed serving-path wins. In particular the
E=32 large-prefill regressions are not dispatched by this branch.

At T=512/E=32 on card 7, record-on routing/reduce took 45.48/2.12 µs and
record-off took 44.74/1.82 µs. The device flag remains graph-safe and
record-off leaves load unchanged, but it still launches both kernels. That
approximately 1 µs difference is an observed single-capture effect, not a
repeatable performance claim or reason to specialize on a host-read flag.

Artifacts are under `/home/shy/moe-gating-topk-map-record/grid-record-v1/`:
`round7_matrix96.log`, `round7_npu_ut.log`, `round7_edge.log`,
`round7_graph.log`, `round7_ir/`, `round7_msprof/`, and the schedule/tile
profile directories. The first measured source was SHA256
`b771531a762488e53facae6c7fb166b2129f63fff0cd98e61bbcc03890643660`.
A formatting-only wrap produced the final source SHA256
`c53364198c3d48f14d43b6af5182cd40560bb709dbca50793f794e9a7e6dfb69`;
their Python ASTs are identical. The final source again passed 22/22 NPU UT,
its TTIR/NPU IR had no atomic, and T=64, T=512 and T=65536 in the table use
its exact-source profiler captures. The final recaptures give 46.94 µs
versus 46.80 µs at T=512 (parity within a single-capture margin), and
2311.74 µs versus 2536.22 µs at T=65536 (8.9% lower task cost). Do not
interpret the smaller margin as a repeatable win.
