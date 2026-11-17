# EPLB all-gather route counting

`count_allgather_tokens_triton(ids, mask, start, count)` implements the NPU path
of `distributed/eplb/tensor_ops.py::count_allgather_tokens`. CPU inputs retain
the existing Torch reference. Inputs are INT32/INT64 physical expert IDs [T,K],
a BOOL token mask [T], and the nonnegative local start and positive count C.
Output is a fresh INT64 [C] tensor: `out[e] = sum(mask[t] and ids[t,j] == start+e)`.
Repeated routes count separately; masked, negative and nonlocal routes count zero.
NPU physical IDs must be contiguous; noncontiguous IDs raise ValueError rather
than being copied implicitly. Strided token masks and empty T/K are supported.

Fixed 512-route tiles are balanced across a grid
bounded by the Vector Core count from the shared capability helper. Each program
accumulates a private INT64 histogram across all its tiles, then writes one row
of a fresh `[programs, num_bins]` INT64 scratch tensor. A second kernel reduces
the program dimension and writes the fresh output without global atomics.
Every scratch and output element used is overwritten per call, including Graph
replay; empty inputs return zeros. Histogram length is the next power of two
of C, with a minimum of two: the A5 backend failed to lower a one-element
loop-carried tensor, including an attempted sum reduction. Padding resolves that
case without changing the 128-bin main case. The caller selects the current NPU,
following the normal worker convention.

IDs use `ids_ptr + offsets` directly. This preserves contiguous access in the
compiler IR instead of reconstructing offsets with division and remainder.

No host reads of tensor values or global intermediate index/mask tensors are
introduced. The 128-bin logical accumulator is 1 KiB; partial-count scratch is
40 KiB for 40 programs or 64 KiB for 64 programs. Final physical UB placement
and compiler-internal workspace have not been established.

The caller still owns gathering/padding, record-enable state and persistent
load accumulation. This is the existing 0/1 route-count contract, not a general
weighted scatter API.

## Validation

### Two-stage candidate (2026-11-17)

A3 and A5 each passed 83/83 exact tests. Mutable Graph coverage includes
20 calls/capture at [10240,8], with repeated changes to IDs and masks.
Same-device full-wrapper Graph medians against atomic commit `a75a38f92`:

| Device | Distribution | Atomic us | Two-stage us |
| --- | --- | ---: | ---: |
| A3 | uniform | 625.29 | 545.24 |
| A3 | hot | 592.72 | 523.85 |
| A3 | mixed | 595.62 | 521.33 |
| A5 | uniform | 1034.54 | 308.09 |
| A5 | hot | 1024.91 | 196.38 |
| A5 | mixed | 1028.70 | 201.18 |

Protocol: 20 complete calls/capture, 5 warmup replays, 10 alternating Event
samples, replay time divided by 20. A5 two-stage timings are unstable: this
paired run ranges from 148.86 to 400.95 us, while a separate four-way diagnostic
also observed roughly 18-us samples. Do not report the lowest sample as stable
performance. The first four-way run failed its final correctness assertion;
that failure is retained and its cause remains unresolved. Subsequent paired
and per-replay-checked four-way runs passed. This remains a draft candidate.

A separate, repeated A5 control keeps the old atomic code and changes only the
program limit to match A3's 40 programs:

| Distribution | 64 programs us | 40 programs us |
| --- | ---: | ---: |
| uniform | 1033.99 | 459.28 |
| hot | 1025.68 | 457.91 |
| mixed | 1027.43 | 457.79 |

This reduction in latency was reproduced in the four-way control. Two-stage
40/64 comparisons remain unstable, so the current candidate keeps the existing
device-based program policy; the 40-program controls are experiments. Removing
atomic also enables A5 AutoBlockify in compiler metadata, so the two-stage gain
cannot be attributed exclusively to atomic contention. No new msprof or model
integration was performed for this candidate. Ruff check/format and diff checks
pass; full format.sh ci is blocked by unavailable pre-commit in the Bash environment.

The final four-way repeat used the original timing synchronization protocol
(no CPU correctness copy between timed replays) and passed pre/post checks:

| Distribution | Atomic64 us | Atomic40 us | Two-stage64 us | Two-stage40 us |
| --- | ---: | ---: | ---: | ---: |
| uniform | 1033.80 | 456.68 | 194.67 | 194.53 |
| hot | 1034.37 | 455.87 | 196.42 | 196.45 |
| mixed | 1033.27 | 456.55 | 193.90 | 194.24 |

These medians do not remove the earlier failure or the within-run timing
outliers; they show no material median advantage for 40 producers once the
global atomic merge is removed in this workload.

### Historical contiguous-ID atomic implementation (2026-11-17)

A3 and A5 each passed 82/82 tests, including noncontiguous-ID rejection and
mutable Graph replay. IDs lower to contiguous `memref.copy` in adapter IR;
token-mask scalar loads remain. This change does not vectorize the entire kernel.
Tile size 512, histogram, grid policy and explicit SIMD mode are unchanged.
Same-device alternating Graph measurements against pre-change commit `814f16c81`
(20 full calls/replay, 10 samples, median microseconds):

| Device | Distribution | Previous us | Contiguous IDs us |
| --- | --- | ---: | ---: |
| A3 | uniform | 651.93 | 611.73 |
| A3 | hot | 641.64 | 595.52 |
| A3 | mixed | 633.70 | 593.70 |
| A5 | uniform | 1060.19 | 1034.15 |
| A5 | hot | 1049.89 | 1024.99 |
| A5 | mixed | 1053.79 | 1029.78 |

The address change improves this measured range but does not explain the full
A3/A5 latency gap. Model-level integration and full repository CI are pending.

### Historical compiler-mode and algorithm trials

The current launch explicitly sets `compile_mode="simd"` on Ascend 950.
A2/A3 already use SIMD under the default hybrid mode. See the
[compiler options reference](https://triton-ascend.readthedocs.io/en/v3.6.0/environment_variable_and_compiler_options_reference.html).
Candidate `a46145c87` passed all 81 correctness tests on Ascend950DT_9582.
Metadata confirms `compile_mode="simd"`, but saved adapter IR still lowers
`tl.histogram` to `__builtin_histogram` with `vf_mode<SIMT>`. This is an IR
classification, not proof of the final instruction path: upstream AscendNPU-IR
revision `fbf6d5ecc72c626100c147f7cf600f5945c04e87` includes both SIMT atomics
and eligible `dhistv2` SIMD paths inside its histogram template. Whether the
installed CANN binary contains those paths has not been established.
Paired, alternating Graph measurements on the same card (20 complete calls per
replay, 10 samples) compare default hybrid versus explicit SIMD:

| Distribution | Default hybrid ms | Explicit SIMD ms | SIMD time increase |
| --- | ---: | ---: | ---: |
| uniform | 1.0299 | 1.0587 | 2.80% |
| hot | 1.0208 | 1.0509 | 2.95% |
| mixed | 1.0246 | 1.0537 | 2.84% |

This option did not improve performance. A3 has not been rerun with the explicit
option. The remaining measurements below belong to commit `980f446c2`, before
this option was added.

A separate custom equality-plus-integer-reduction experiment (128 routes/tile,
int64 per-program accumulation, final atomic merge) passed 81/81 A5 tests.
Its 11 saved adapter variants contain no histogram builtin or SIMT VF operation
and report `parallel_mode="simd"`. Same-card paired Graph timings were:

| Distribution | Default histogram ms | Custom reduction ms | Time increase |
| --- | ---: | ---: | ---: |
| uniform | 1.0301 | 1.0707 | 3.95% |
| hot | 1.0207 | 1.0753 | 5.34% |
| mixed | 1.0233 | 1.0761 | 5.16% |

The custom implementation was not promoted into this PR kernel because it
regressed all three measured cases. This experiment changed both the counting
method and tile size; it does not isolate the hardware SIMD/SIMT throughput.

Both A3 Ascend910_9362 (NpuArch 2201) and A5 Ascend950DT_9582 (NpuArch 3510)
passed 81 tests with exact integer equality, including mutable-input Graph replay.
Cases include both ID dtypes, strides, empty inputs, single expert, 511/512/513
route boundaries, non-power-of-two expert counts, hotspots, disabled tokens,
nonlocal and large INT64 IDs. The kernel, wrapper and helper import hashes match
the tested source. The package snapshot was taken from this PR and the final
minimum-two-bin correction overlaid before both device runs.

Both environments: CANN 9.2.1, Python 3.12.16, Torch 2.10.0+cpu,
Torch-NPU 2.10.0.post7, Triton 3.6.0 / Triton-Ascend 3.6.0.post2.
A3 driver: 26.2.rc1.b081; A5 driver: 25.6.rc3.1.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb_count.py
```

For standalone validation, the test file was copied outside the e2e tree to avoid
unrelated conftest setup, with the tested package first on PYTHONPATH.

## Complete-function Graph performance

Synthetic INT64 IDs [10240,8], output [128], local start 7. Each Graph captures
20 complete count calls; after 5 warmup replays, 10 samples alternate original
Torch/candidate execution order. NPU Event replay duration is divided by 20;
medians are below. Outputs are checked before and after measurement.

| Device | Distribution | Torch ms | Histogram ms | Torch / histogram |
| --- | --- | ---: | ---: | ---: |
| A3 | uniform | 0.6136 | 0.6401 | 0.959x |
| A3 | hot | 0.7204 | 0.6174 | 1.167x |
| A3 | mixed | 0.6192 | 0.6466 | 0.958x |
| A5 | uniform | 2.2008 | 1.0258 | 2.145x |
| A5 | hot | 13.9194 | 1.0270 | 13.554x |
| A5 | mixed | 2.3428 | 1.0272 | 2.281x |

Uniform IDs cycle start+[0,127]; hot IDs equal start; mixed IDs cycle
[start-2,start+129], with every third token disabled. These are synthetic operator
results, not production-model speedups. A3 independent repeat speedups were
0.923x / 1.136x / 0.968x (uniform / hot / mixed). Its non-hot cases regressed
in both runs, by about 3-8.3% in latency. This is not an unconditional A3 win.

Separate full msprof uniform runs retained 20 calls after 5 warmups. Original
Torch uses 9 compute kernels/call; the histogram path uses 2. Mean counting-kernel
durations were 730.851 / 664.864 us (Torch / histogram) on A3 and
2140.067 / 1002.915 us on A5. These eager task durations are distinct from the
Graph complete-function timings above. Model/serving validation and repository
CI remain pending; the PR stays draft for review of the A3 tradeoff.
