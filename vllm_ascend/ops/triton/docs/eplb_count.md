# EPLB all-gather route counting

## Description

- **Function:** Count valid physical routes owned by local experts. The shared
  `distributed/eplb/tensor_ops.py::count_allgather_tokens` wrapper dispatches
  from the all-gather recording path in `eplb_state.py`.
- **Formula:** `out[e] = sum(mask[t] and ids[t,j] == start + e)`.
  Duplicate routes count separately; masked and nonlocal routes count zero.
- **Algorithm:** Two Triton launches. Producers balance fixed 512-route tiles
  over `min(vector_cores, ceil(T*K/512))` programs, accumulate private INT64
  histograms and write complete scratch rows. Producer 0 also zeros the output.
  Each program in the second kernel reads one adjacent row tile, reduces axis
  0 and atomically adds its INT64 vector to the output. There is no row loop
  in the second kernel. Empty T/K returns zeros.
- **Platform:** Ascend NPU using the worker-selected device and the repository's
  shared vector-core capability helper. Current validation is reported below.

## Parameters

Public kernel wrapper:
`vllm_ascend.ops.triton.eplb_count.count_allgather_tokens_triton`.

| Parameter | Input/Output/Attribute | Description | Data type | Format |
| --- | --- | --- | --- | --- |
| `topk_ids` | Input | Physical expert IDs | INT32/INT64 | Contiguous `[T,K]` |
| `token_mask` | Input | Whether each token contributes | BOOL | `[T]`, strides supported |
| `local_expert_start` | Attribute | First local physical expert | Python int | Nonnegative scalar |
| `local_expert_count` | Attribute | Number of local experts | Python int | Positive scalar |
| return value | Output | Fresh local route counts | INT64 | `[local_expert_count]` |

## Constraints

IDs and mask must be on the same NPU with matching T. Noncontiguous IDs are
rejected without an implicit copy. IDs use `ids_ptr + offsets`; masks may be
strided. Route count and expert bounds use the backend's INT32 scalar domain;
IDs and accumulated counts use INT64. Out-of-range IDs are ignored.

`padded_bins = max(2, next_power_of_2(local_expert_count))`. The minimum of two
keeps the producer's loop-carried histogram vector-shaped for the backend.
The producer tile has 512 routes on both devices. The reducer chooses
`rows_per_tile = min(programs, max(1, 8192 // padded_bins))` and pads the row
dimension to a power of two for the shaped load. Each reducer program loads
exactly one row tile, with tail rows masked, and uses a vector atomic add.

The budget is 8192 INT64 partial counts (64 KiB); reduction and compiler
temporaries use additional UB. For the validated expert counts the padded
load stays within this element budget. Validation covers up to 512 experts.

| Padded bins | A3 rows/tile; reducer programs (40 producers) | A5 rows/tile; reducer programs (64 producers) |
| ---: | --- | --- |
| 128 | 40; 1 | 64; 1 |
| 256 | 32; 2 | 32; 2 |
| 512 | 16; 3 | 16; 4 |

Scratch takes `8 * programs * padded_bins` bytes: 40 KiB for 40 producers and
128 bins, or 64 KiB for 64 producers. These are logical allocation sizes.
Output and scratch are fresh per call. The two launches use the same current
stream: completion of the first kernel orders both output zeroing and scratch
writes before all second-kernel atomic updates. There is no extra clear launch.
Graph replay re-executes this zeroing on every call. Inputs are not mutated,
and no tensor values are read on the host. The existing caller still owns
all-gather, record-enable state and persistent load accumulation.

## Origin and Differences

The origin is the shared count contract in main
`vllm_ascend/distributed/eplb/tensor_ops.py`. Main constructs local IDs,
eligibility and clamped indices, zeros the output and invokes Torch
`scatter_add_`. This implementation combines eligibility and local histogram
construction, then merges partial counts in a separate kernel. CPU inputs
retain main's Torch path. NPU IDs must be contiguous; masks may be strided.

### Compilation and integration

Production callers use the public wrapper. Producer variants depend on pointer
dtype/alignment, TOPK, mask stride, padded bins, tile and target/compiler options.
Route count and expert start/count use `do_not_specialize`. Reducer variants
depend on producer count, padded bins, row tile and valid expert count. Changing
only tensor contents does not create a new variant.

Both launches request `compile_mode="simd"`; histogram lowering on A5 can still
generate mixed SIMD/SIMT code. No warmup registration or new runtime setting is
introduced. Model startup/serving compilation counts remain unmeasured.

## Test Cases

Reference: main's Torch formula on CPU. INT64 outputs require exact equality
(`rtol=0`, `atol=0`). There are 455 parameterized tests: 45 shapes x two ID
dtypes x five distributions, one noncontiguous-ID rejection and four mutable
Graph regressions. Coverage includes empty T/K, 511/512/513-route boundaries,
40/64-producer transitions, non-power-of-two bins, up to 1,048,576 routes and
512 experts, large/nonlocal and duplicate IDs, strided/disabled masks, and
15/16/17 partial-row boundaries. The mutable Graphs include 20-call capture
with multiple reducer programs and repeated ID/mask changes.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb_count.py
```

The expanded shapes are synthetic interface/boundary cases. The task's original
81920-route/128-bin workload is included; added shapes are not model captures.

## Performance

The measured candidate is `dda67fd5e`, merged through main `776d8f084`.
The baseline is main `3ae9de590`; its `tensor_ops.py` is byte-identical in
`776d8f084`. Kernel SHA256:
`578169f556f9f8baa549c711e7e6332ec45e2b97038e2eca5f4600981ab3f948`.

Graph-only measurement covers the complete public wrapper, including fresh
output/scratch allocation in capture. Each Graph captures 20 calls, performs
five warmup replays and collects ten alternating device Event samples; values
are median replay time divided by 20. Both implementations are checked against
the CPU oracle before capture and after replay. There is no eager timing.

The matrix has 43 nonempty shapes x three distributions (uniform, hot, mixed),
three extra INT32 shapes x three distributions, and one strided-mask shape x
three distributions: 141 points/device. All failures and regressions are retained.

| Platform | Exact tests | Graph exact checks | Faster medians | Regressions | Speedup range |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3 | 455/455 | 141/141 | 117/141 | 24/141 | 0.581x–1.983x |
| A5 | NOT RUN | NOT RUN | — | — | — |

A3 uses Ascend910_9362 / NpuArch2201, 40 vector cores and driver 26.2.rc1.b081.
The target was healthy and no other process on that target was observed in
ownership checks; other host devices had active workloads. This is a shared
host measurement, not a reservation. CANN is 9.2.1, Torch 2.10.0+cpu, Torch-NPU
2.10.0.post7, Triton 3.6.0 and Triton-Ascend 3.6.0.post2. The compiler SHA256 is
`69b9402f711ccdcd21c6f951b5e211bb2abd397f1f5ff06dd4dc7a88c6f974bc`.

Current A5 validation is NOT RUN: replacement targets were occupied or the
isolated runtime failed initialization with DCMI `ret=-8020` before a kernel
test. A5 rows contain null timing data and NOT_RUN status. Earlier candidate
measurements are not validation of this source.

For INT64 IDs, contiguous masks and uniform local routes, shapes are
`(T,K,local_expert_count)` and times are microseconds per call:

| Shape | A3 main | A3 two-stage | A3 speedup | A5 main | A5 two-stage | A5 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 1, 1) | 43.22 | 71.23 | 0.607x | — | — | — |
| (1, 8, 128) | 101.11 | 76.73 | 1.318x | — | — | — |
| (32, 2, 8) | 90.77 | 73.87 | 1.229x | — | — | — |
| (64, 4, 32) | 105.47 | 84.35 | 1.250x | — | — | — |
| (65, 8, 17) | 107.68 | 94.79 | 1.136x | — | — | — |
| (128, 3, 64) | 107.45 | 90.48 | 1.188x | — | — | — |
| (256, 8, 64) | 127.89 | 101.65 | 1.258x | — | — | — |
| (511, 1, 128) | 115.73 | 93.56 | 1.237x | — | — | — |
| (512, 1, 128) | 111.03 | 93.69 | 1.185x | — | — | — |
| (513, 1, 129) | 116.13 | 106.21 | 1.093x | — | — | — |
| (512, 8, 128) | 146.43 | 120.94 | 1.211x | — | — | — |
| (2048, 8, 128) | 312.69 | 191.16 | 1.636x | — | — | — |
| (10240, 1, 128) | 183.81 | 187.76 | 0.979x | — | — | — |
| (10240, 4, 128) | 510.24 | 374.33 | 1.363x | — | — | — |
| (10240, 8, 32) | 591.43 | 403.70 | 1.465x | — | — | — |
| (10240, 8, 64) | 603.45 | 453.40 | 1.331x | — | — | — |
| (10240, 8, 128) | 614.89 | 550.74 | 1.116x | — | — | — |
| (10240, 8, 256) | 619.46 | 568.17 | 1.090x | — | — | — |
| (10240, 16, 128) | 1418.81 | 905.81 | 1.566x | — | — | — |
| (32768, 8, 128) | 2186.41 | 1350.03 | 1.620x | — | — | — |
| (2, 1, 1) | 44.62 | 72.26 | 0.618x | — | — | — |
| (16, 2, 16) | 93.68 | 73.44 | 1.276x | — | — | — |
| (127, 8, 127) | 121.52 | 103.29 | 1.176x | — | — | — |
| (129, 8, 129) | 121.38 | 122.09 | 0.994x | — | — | — |
| (1024, 2, 128) | 131.23 | 108.19 | 1.213x | — | — | — |
| (1024, 8, 128) | 209.27 | 144.47 | 1.449x | — | — | — |
| (2559, 8, 128) | 277.64 | 285.37 | 0.973x | — | — | — |
| (2560, 8, 128) | 284.01 | 285.27 | 0.996x | — | — | — |
| (2561, 8, 128) | 271.60 | 370.24 | 0.734x | — | — | — |
| (4095, 8, 128) | 394.65 | 372.00 | 1.061x | — | — | — |
| (4096, 8, 128) | 447.40 | 372.02 | 1.203x | — | — | — |
| (4097, 8, 128) | 451.43 | 372.07 | 1.213x | — | — | — |
| (8192, 8, 128) | 684.16 | 548.49 | 1.247x | — | — | — |
| (16384, 8, 128) | 1176.30 | 815.61 | 1.442x | — | — | — |
| (65536, 8, 128) | 4283.53 | 2505.28 | 1.710x | — | — | — |
| (131072, 8, 128) | 6364.25 | 4813.39 | 1.322x | — | — | — |
| (10240, 2, 128) | 284.27 | 284.38 | 1.000x | — | — | — |
| (10240, 32, 128) | 2702.06 | 1616.57 | 1.671x | — | — | — |
| (10240, 8, 16) | 585.73 | 377.40 | 1.552x | — | — | — |
| (10240, 8, 512) | 624.90 | 611.39 | 1.022x | — | — | — |
| (960, 8, 512) | 176.56 | 297.37 | 0.594x | — | — | — |
| (1024, 8, 512) | 211.46 | 294.74 | 0.717x | — | — | — |
| (1088, 8, 512) | 186.72 | 306.51 | 0.609x | — | — | — |

The original 81920-route/128-bin workload, with local start 7:

| Device / distribution | Main (us) | Two-stage (us) | Speedup |
| --- | ---: | ---: | ---: |
| A3 / uniform | 614.89 | 550.74 | 1.116x |
| A3 / hot | 717.44 | 529.34 | 1.355x |
| A3 / mixed | 619.11 | 527.73 | 1.173x |

A3 has 24 regressions. Its lowest ratio is 0.581x at `(1, 1, 1)`, `mixed`, int64, mask stride 1. The implementation is not faster than main at every shape. One reducer tile still pays the output zeroing, INT64 reduction and atomic merge costs; these costs can outweigh the smaller input scan.

Every point, status, median, observed min/max, ten raw samples per path and
source identity is in the [A3 CSV](eplb_count_graph_a3.csv) and
[A5 CSV](eplb_count_graph_a5.csv). Missing results have empty numeric fields.
Model accuracy, serving performance and startup/serving compilation counts
have not been measured by these standalone tests. An earlier A5 multi-Graph
failure has no established root cause and is retained in the local history.

### Reproduce

Extract the pinned main baseline, then run the committed benchmark from the
repository root with the same NPU dependencies:

```bash
mkdir -p /tmp/eplb-main-baseline
git show 3ae9de5900e181817ad86a0fa936c379a3789c0f:vllm_ascend/distributed/eplb/tensor_ops.py > /tmp/eplb-main-baseline/baseline_tensor_ops.py
PYTHONPATH="$PWD:/tmp/eplb-main-baseline:$PYTHONPATH" python benchmarks/eplb_count_graph.py
```

The runner writes every point and raw sample to `results/expanded-graph.json`,
including errors with null timing fields, and saves imported paths and source
hashes in `results/import-identity.json`. Use separate run directories/devices.
