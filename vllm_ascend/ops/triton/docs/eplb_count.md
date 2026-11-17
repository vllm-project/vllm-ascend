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

The measured candidate is `744847056`, merged through main `776d8f084`.
The baseline is main `3ae9de590`; its `tensor_ops.py` is byte-identical in
`776d8f084`. Kernel SHA256:
`8ee968a3079ae1f52a627a6634e5f4fed595c84b27394d0d293fc10bf55f2818`.

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
| A3 | 455/455 | 141/141 | 117/141 | 24/141 | 0.574x–1.953x |
| A5 | 455/455 | 141/141 | 118/141 | 23/141 | 0.237x–910.690x |

A3 and A5 use CANN 9.2.1, Torch 2.10.0+cpu, Torch-NPU 2.10.0.post7,
Triton 3.6.0 and Triton-Ascend 3.6.0.post2. The Triton compiler binary identity
matches across both platforms. These are paired main-versus-candidate
measurements in shared test environments, with periodic target-process checks.
They do not establish an isolated comparison of A3 and A5 hardware.

For INT64 IDs, contiguous masks and uniform local routes, shapes are
`(T,K,local_expert_count)` and times are microseconds per call:

| Shape | A3 main | A3 two-stage | A3 speedup | A5 main | A5 two-stage | A5 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 1, 1) | 42.62 | 71.22 | 0.598x | 10.46 | 35.57 | 0.294x |
| (1, 8, 128) | 102.12 | 76.65 | 1.332x | 13.86 | 35.86 | 0.387x |
| (32, 2, 8) | 91.01 | 73.95 | 1.231x | 27.92 | 38.87 | 0.718x |
| (64, 4, 32) | 105.50 | 83.29 | 1.267x | 44.12 | 37.59 | 1.174x |
| (65, 8, 17) | 104.72 | 94.85 | 1.104x | 146.16 | 50.72 | 2.882x |
| (128, 3, 64) | 105.56 | 90.40 | 1.168x | 35.11 | 38.64 | 0.909x |
| (256, 8, 64) | 124.53 | 100.61 | 1.238x | 135.43 | 39.59 | 3.421x |
| (511, 1, 128) | 115.25 | 93.55 | 1.232x | 28.85 | 35.59 | 0.811x |
| (512, 1, 128) | 112.27 | 93.76 | 1.197x | 28.55 | 35.51 | 0.804x |
| (513, 1, 129) | 115.74 | 105.07 | 1.102x | 29.08 | 35.74 | 0.814x |
| (512, 8, 128) | 144.99 | 119.81 | 1.210x | 140.64 | 40.05 | 3.512x |
| (2048, 8, 128) | 305.60 | 190.30 | 1.606x | 453.31 | 41.33 | 10.968x |
| (10240, 1, 128) | 180.33 | 186.59 | 0.966x | 288.13 | 36.84 | 7.822x |
| (10240, 4, 128) | 499.65 | 373.37 | 1.338x | 1094.25 | 47.35 | 23.108x |
| (10240, 8, 32) | 591.23 | 404.08 | 1.463x | 7642.23 | 50.28 | 151.989x |
| (10240, 8, 64) | 595.53 | 453.23 | 1.314x | 4358.33 | 53.46 | 81.520x |
| (10240, 8, 128) | 611.73 | 551.08 | 1.110x | 2195.50 | 51.50 | 42.635x |
| (10240, 8, 256) | 619.00 | 568.18 | 1.089x | 1195.69 | 67.07 | 17.827x |
| (10240, 16, 128) | 1413.95 | 905.93 | 1.561x | 4336.93 | 60.64 | 71.518x |
| (32768, 8, 128) | 2151.83 | 1349.65 | 1.594x | 6925.28 | 84.41 | 82.039x |
| (2, 1, 1) | 43.52 | 71.28 | 0.611x | 10.70 | 45.22 | 0.237x |
| (16, 2, 16) | 88.83 | 72.48 | 1.226x | 19.38 | 35.75 | 0.542x |
| (127, 8, 127) | 120.38 | 102.68 | 1.172x | 41.98 | 39.96 | 1.050x |
| (129, 8, 129) | 121.02 | 121.14 | 0.999x | 41.44 | 40.05 | 1.035x |
| (1024, 2, 128) | 132.53 | 108.29 | 1.224x | 75.45 | 39.90 | 1.891x |
| (1024, 8, 128) | 207.81 | 143.52 | 1.448x | 238.35 | 40.31 | 5.912x |
| (2559, 8, 128) | 274.60 | 284.23 | 0.966x | 552.52 | 42.37 | 13.041x |
| (2560, 8, 128) | 279.08 | 284.55 | 0.981x | 560.77 | 42.48 | 13.200x |
| (2561, 8, 128) | 270.02 | 370.25 | 0.729x | 564.13 | 43.05 | 13.103x |
| (4095, 8, 128) | 392.16 | 372.09 | 1.054x | 892.73 | 42.88 | 20.819x |
| (4096, 8, 128) | 444.09 | 372.29 | 1.193x | 893.61 | 43.08 | 20.743x |
| (4097, 8, 128) | 451.36 | 372.11 | 1.213x | 881.59 | 57.48 | 15.338x |
| (8192, 8, 128) | 683.45 | 548.58 | 1.246x | 1741.42 | 47.69 | 36.518x |
| (16384, 8, 128) | 1155.31 | 815.50 | 1.417x | 3449.59 | 56.33 | 61.236x |
| (65536, 8, 128) | 4243.71 | 2504.06 | 1.695x | 13757.96 | 120.21 | 114.452x |
| (131072, 8, 128) | 6288.70 | 4812.46 | 1.307x | 27441.73 | 185.47 | 147.958x |
| (10240, 2, 128) | 281.56 | 284.75 | 0.989x | 552.47 | 42.99 | 12.851x |
| (10240, 32, 128) | 2655.72 | 1615.86 | 1.644x | 8596.93 | 93.20 | 92.241x |
| (10240, 8, 16) | 587.28 | 377.30 | 1.557x | 14301.91 | 55.81 | 256.277x |
| (10240, 8, 512) | 618.46 | 610.65 | 1.013x | 842.63 | 61.52 | 13.696x |
| (960, 8, 512) | 174.15 | 296.33 | 0.588x | 121.08 | 43.59 | 2.778x |
| (1024, 8, 512) | 209.93 | 293.97 | 0.714x | 131.68 | 43.30 | 3.041x |
| (1088, 8, 512) | 184.88 | 305.57 | 0.605x | 134.18 | 46.82 | 2.866x |

The original 81920-route/128-bin workload, with local start 7:

| Device / distribution | Main (us) | Two-stage (us) | Speedup |
| --- | ---: | ---: | ---: |
| A3 / uniform | 611.73 | 551.08 | 1.110x |
| A3 / hot | 716.41 | 529.54 | 1.353x |
| A3 / mixed | 613.08 | 526.83 | 1.164x |
| A5 / uniform | 2195.50 | 51.50 | 42.635x |
| A5 / hot | 13636.53 | 62.90 | 216.797x |
| A5 / mixed | 2317.06 | 51.95 | 44.600x |

A3 has 24 regressions. Its lowest ratio is 0.574x at `(2, 1, 1)`, `mixed`, int64, mask stride 1. The implementation is not faster than main at every shape. One reducer tile still pays the output zeroing, INT64 reduction and atomic merge costs; these costs can outweigh the smaller input scan.

The regressions include tiny inputs, 512-bin inputs with only 15/16/17 partial
rows, and the 40-to-41 producer-tile transition. At `(2560,8,128)`, 40 programs
each scan one 512-route tile. At `(2561,8,128)`, one program scans two tiles;
its second histogram still has 512 lanes for eight valid routes. This imbalance
creates a latency step. Splitting reducer tiles does not remove that producer
cost, and the 15/16-row cases still have only one reducer program.

A5 also has 23 regressions. Its lowest ratio is 0.237x at `(2, 1, 1)`, `uniform`, int64, mask stride 1. Large-shape gains do not establish an all-shape speedup.

Every point, status, median, observed min/max, ten raw samples per path and
source identity is in the [A3 CSV](eplb_count_graph_a3.csv) and
[A5 CSV](eplb_count_graph_a5.csv). Missing results have empty numeric fields.
Model accuracy, serving performance and startup/serving compilation counts
have not been measured by these standalone tests.

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
