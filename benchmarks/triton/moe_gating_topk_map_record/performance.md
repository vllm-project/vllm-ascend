# MoE gating TopK + EPLB mapping/recording performance

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
