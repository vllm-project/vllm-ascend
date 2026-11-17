# EPLB all-gather route counting

## Description

- **Function:** Count valid physical routes owned by local experts. The shared
  `distributed/eplb/tensor_ops.py::count_allgather_tokens` wrapper dispatches
  from the all-gather recording path in `eplb_state.py`.
- **Formula:** `out[e] = sum(mask[t] and ids[t,j] == start + e)`.
  Duplicate routes count separately; masked and nonlocal routes count zero.
- **Algorithm:** Two Triton launches. Producers balance fixed 512-route tiles,
  accumulate private histograms and write complete scratch rows. Each reducer
  program loads one adjacent row tile and reduces axis 0. A single reducer
  overwrites the output; multiple reducers merge their INT64 vectors atomically
  after producer 0 initializes the output. Empty T/K returns zeros.
- **Platform:** Ascend NPU using the worker-selected device and the repository's
  shared vector-core capability helper. A3 and A5 validation is reported below.

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
strided. `K` maps each route to its token mask through `offset // K`; counting
uses the flattened route stream. IDs and expert-range comparisons use INT64.
Out-of-range IDs are ignored.

Each route contributes at most one. When `T*K <= 2**31 - 1`, every local,
row-tile and final count therefore fits signed INT32: producer accumulators,
scratch and the row reduction use INT32, then the result converts to INT64.
Larger route counts select INT64 scratch and accumulation. The fallback is
tested by forcing its dispatch on a small input and by reducing synthetic
INT64 partial counts above `2**31`; an input with more than `2**31 - 1` routes
has not been allocated in these tests. The returned dtype is always INT64.

`programs = min(vector_cores, ceil(T*K/512))` and
`padded_bins = max(2, next_power_of_2(local_expert_count))`. The minimum of two
keeps the producer's loop-carried histogram vector-shaped for the backend.
Fixed 512-route tiles are distributed by quotient and remainder, so producer
tile counts differ by at most one. Histogram eligibility uses its mask argument.

The reducer chooses `rows_per_tile = min(programs, max(1, 8192 // padded_bins))`
and pads rows to a power of two. Each program loads one row tile, with tail
rows masked, and reduces the producer dimension. There is no reducer row loop.
The 8192-element budget uses at most 32 KiB for INT32 or 64 KiB for INT64
partials in the validated expert range; compiler temporaries use additional UB.
Validation covers up to 512 experts and 1,048,576 routes.

| Padded bins | A3 rows/tile; reducer programs (40 producers) | A5 rows/tile; reducer programs (64 producers) |
| ---: | --- | --- |
| 128 | 40; 1, direct store | 64; 1, direct store |
| 256 | 32; 2, atomic merge | 32; 2, atomic merge |
| 512 | 16; 3, atomic merge | 16; 4, atomic merge |

Logical scratch allocation is `sizeof(counter) * programs * padded_bins`:
for 81920 routes and 128 bins, INT32 scratch is 20 KiB on A3 and 32 KiB on A5.
Output and scratch are fresh per call. With one reducer, it owns and overwrites
every output element, so output zeroing is omitted. With multiple reducers,
producer 0 zeros the output and the two launches' current-stream dependency
orders initialization and scratch writes before all atomic updates. Graph
replay repeats the required writes on every call. No extra clear kernel is used.
Inputs are not mutated and no tensor values are read on the host. The existing
caller owns all-gather, record-enable state and persistent load accumulation.

## Origin and Differences

The origin is the shared count contract in main
`vllm_ascend/distributed/eplb/tensor_ops.py`. Main constructs local IDs,
eligibility and clamped indices, zeros the output and invokes Torch
`scatter_add_`. This implementation combines eligibility and local histogram
construction, then reduces partial counts in a separate kernel. CPU inputs
retain main's Torch path. NPU IDs must be contiguous; masks may be strided.

### Compilation and integration

Production callers use the public wrapper. Producer variants depend on pointer
dtype/alignment, TOPK, mask stride, padded bins, tile, output ownership and
target/compiler options. Route count and expert start/count use
`do_not_specialize`. Scratch pointer dtype selects the counter-width family;
no additional counter-width constexpr is needed. Reducer variants depend on
producer count, padded bins, row tile, valid expert count and output ownership.
`NUM_PARTIAL_ROWS` names the valid scratch-row count, not the reducer grid size.
Changing only tensor contents does not create a new variant.

Both launches request `compile_mode="simd"`; histogram lowering on A5 can still
generate mixed SIMD/SIMT code. Current TTIR and adapter IR show an INT32
axis-0 reduction, sign extension to INT64 and direct store for the 128-bin
single-reducer case; multi-reducer variants retain atomic merging. Reducer
TTIR contains no `scf.for`. These observations describe intermediate IR.
No warmup registration or new runtime setting is introduced. Model
startup/serving compilation counts remain unmeasured.

## Test Cases

Reference: main's Torch formula on CPU. INT64 outputs require exact equality
(`rtol=0`, `atol=0`). There are 468 tests: 46 shapes x two ID dtypes x five
distributions, one noncontiguous-ID rejection, four mutable Graph regressions,
two large-INT64 reducer checks and one forced INT64 full-wrapper fallback.
Coverage includes empty T/K, 511/512/513-route boundaries, 40/64-producer
transitions, 15/16/17 partial-row boundaries, non-power-of-two bins, up to
1,048,576 routes and 512 experts, large/nonlocal and duplicate IDs, and
strided/disabled masks. Mutable Graphs include 20-call captures with single
and multiple reducer programs and repeated ID/mask changes.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb_count.py
```

Shapes are synthetic interface/boundary cases. The task's 81920-route/128-bin
workload includes both `[10240,8]` and `[81920,1]` decompositions. They have
equivalent counts when flattened IDs and per-route eligibility match. Actual
model T/K is not inferred from the flattened scatter shape.

## Performance

Code commit: `30faa0108cfa16a88d87e9430b92dface435cf13`. Measured checkout: `7f87d4039321d3205f8406363f930b621c0e3aec`
(an additional measurement-only runner; kernel, public wrapper, tests and
expanded benchmark match the code commit). Main is merged through
`776d8f084a46f2c00dffacf56ccd25dad57a5c62`. Baseline main: `3ae9de5900e181817ad86a0fa936c379a3789c0f`;
its `tensor_ops.py` is byte-identical in that merged main. Kernel SHA256:
`498f17f4d8d51cee744cf791032504bcf2deaea78838d1676c11d29016f263ef`.

Graph-only measurement covers the complete public wrapper, with fresh
output/scratch allocated during capture. Each Graph captures 20 calls,
performs five warmup replays and collects ten alternating device Event
samples; values are median replay time divided by 20. Both implementations
are checked against the CPU oracle before capture and after replay. There
is no eager timing.

The matrix has 44 nonempty shapes x three distributions (uniform, hot, mixed),
three extra INT32 shapes x three distributions, and one strided-mask shape x
three distributions: 144 points/device. All regressions are retained.

| Platform | Exact tests | Graph exact checks | Faster medians | Regressions | Speedup range |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3 | 468/468 | 144/144 | 138/144 | 6/144 | 0.606x–3.696x |
| A5 | 468/468 | 144/144 | 140/144 | 4/144 | 0.697x–1232.577x |

A3 and A5 use CANN 9.2.1, Torch 2.10.0+cpu, Torch-NPU 2.10.0.post7,
Triton 3.6.0 and Triton-Ascend 3.6.0.post2. The Triton compiler binary identity
matches across both platforms. These are paired main-versus-candidate
measurements in shared test environments with periodic target-process checks.
A5 samples have visible dispersion, retained in the CSV. These runs do not
establish an isolated comparison of A3 and A5 hardware.

For INT64 IDs, contiguous masks and uniform local routes, shapes are
`(T,K,local_expert_count)` and times are microseconds per call:

| Shape | A3 main | A3 two-stage | A3 speedup | A5 main | A5 two-stage | A5 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 1, 1) | 43.14 | 67.71 | 0.637x | 10.45 | 3.07 | 3.411x |
| (1, 8, 128) | 102.85 | 67.79 | 1.517x | 12.24 | 2.79 | 4.387x |
| (32, 2, 8) | 91.14 | 70.05 | 1.301x | 27.15 | 3.30 | 8.226x |
| (64, 4, 32) | 103.73 | 78.02 | 1.329x | 45.78 | 5.19 | 8.829x |
| (65, 8, 17) | 104.98 | 88.85 | 1.181x | 130.38 | 15.67 | 8.319x |
| (128, 3, 64) | 107.69 | 84.25 | 1.278x | 38.96 | 14.60 | 2.670x |
| (256, 8, 64) | 128.21 | 88.94 | 1.441x | 133.51 | 7.54 | 17.705x |
| (511, 1, 128) | 116.40 | 85.33 | 1.364x | 29.20 | 15.69 | 1.861x |
| (512, 1, 128) | 112.82 | 85.43 | 1.321x | 28.78 | 25.48 | 1.129x |
| (513, 1, 129) | 116.09 | 85.50 | 1.358x | 29.47 | 3.38 | 8.723x |
| (512, 8, 128) | 146.75 | 89.47 | 1.640x | 140.17 | 23.16 | 6.053x |
| (2048, 8, 128) | 305.86 | 91.33 | 3.349x | 453.39 | 24.94 | 18.178x |
| (10240, 1, 128) | 184.07 | 86.58 | 2.126x | 288.02 | 4.38 | 65.774x |
| (10240, 4, 128) | 500.21 | 178.08 | 2.809x | 1104.95 | 27.29 | 40.495x |
| (10240, 8, 32) | 585.62 | 349.80 | 1.674x | 7625.56 | 20.69 | 368.640x |
| (10240, 8, 64) | 592.16 | 349.65 | 1.694x | 4350.19 | 23.42 | 185.760x |
| (10240, 8, 128) | 610.76 | 350.15 | 1.744x | 2200.16 | 29.18 | 75.406x |
| (81920, 1, 128) | 598.40 | 334.89 | 1.787x | 2191.30 | 15.42 | 142.129x |
| (10240, 8, 256) | 615.51 | 365.26 | 1.685x | 1188.71 | 53.93 | 22.041x |
| (10240, 16, 128) | 1401.62 | 693.60 | 2.021x | 4313.92 | 27.76 | 155.425x |
| (32768, 8, 128) | 2149.99 | 1124.06 | 1.913x | 6893.53 | 51.82 | 133.039x |
| (2, 1, 1) | 43.35 | 67.76 | 0.640x | 15.03 | 21.57 | 0.697x |
| (16, 2, 16) | 91.62 | 68.56 | 1.336x | 19.10 | 4.01 | 4.760x |
| (127, 8, 127) | 119.89 | 89.09 | 1.346x | 40.67 | 17.06 | 2.384x |
| (129, 8, 129) | 120.29 | 88.87 | 1.353x | 41.78 | 7.42 | 5.631x |
| (1024, 2, 128) | 132.64 | 89.25 | 1.486x | 74.68 | 14.24 | 5.244x |
| (1024, 8, 128) | 206.96 | 90.01 | 2.299x | 238.62 | 15.36 | 15.536x |
| (2559, 8, 128) | 273.04 | 92.06 | 2.966x | 555.01 | 15.05 | 36.877x |
| (2560, 8, 128) | 278.83 | 91.70 | 3.041x | 555.24 | 20.14 | 27.574x |
| (2561, 8, 128) | 269.36 | 174.97 | 1.539x | 560.62 | 15.76 | 35.576x |
| (4095, 8, 128) | 392.98 | 176.90 | 2.221x | 881.57 | 16.26 | 54.202x |
| (4096, 8, 128) | 436.76 | 176.71 | 2.472x | 889.75 | 22.36 | 39.797x |
| (4097, 8, 128) | 444.79 | 176.96 | 2.514x | 888.09 | 20.35 | 43.638x |
| (8192, 8, 128) | 680.01 | 347.62 | 1.956x | 1761.76 | 20.54 | 85.788x |
| (16384, 8, 128) | 1154.02 | 606.46 | 1.903x | 3450.03 | 34.22 | 100.820x |
| (65536, 8, 128) | 4254.73 | 2241.68 | 1.898x | 13809.15 | 87.58 | 157.676x |
| (131072, 8, 128) | 6326.07 | 4477.83 | 1.413x | 27768.19 | 157.88 | 175.884x |
| (10240, 2, 128) | 278.94 | 92.22 | 3.025x | 554.91 | 22.40 | 24.777x |
| (10240, 32, 128) | 2674.59 | 1381.40 | 1.936x | 8594.35 | 54.73 | 157.037x |
| (10240, 8, 16) | 582.34 | 349.34 | 1.667x | 14074.88 | 24.89 | 565.502x |
| (10240, 8, 512) | 616.48 | 387.96 | 1.589x | 846.71 | 68.58 | 12.346x |
| (960, 8, 512) | 176.63 | 90.24 | 1.957x | 120.89 | 15.38 | 7.860x |
| (1024, 8, 512) | 210.75 | 90.20 | 2.336x | 126.16 | 15.36 | 8.211x |
| (1088, 8, 512) | 185.53 | 116.70 | 1.590x | 132.80 | 44.20 | 3.005x |

For the original 81920-route/128-bin workload, local start 7, with two tested
T/K decompositions:

| Platform / shape / distribution | Main (us) | Two-stage (us) | Speedup |
| --- | ---: | ---: | ---: |
| A3 / (10240, 8) / uniform | 610.76 | 350.15 | 1.744x |
| A3 / (10240, 8) / hot | 716.24 | 328.75 | 2.179x |
| A3 / (10240, 8) / mixed | 608.97 | 325.40 | 1.871x |
| A3 / (81920, 1) / uniform | 598.40 | 334.89 | 1.787x |
| A3 / (81920, 1) / hot | 702.44 | 313.41 | 2.241x |
| A3 / (81920, 1) / mixed | 599.04 | 302.20 | 1.982x |
| A5 / (10240, 8) / uniform | 2200.16 | 29.18 | 75.406x |
| A5 / (10240, 8) / hot | 13777.35 | 29.22 | 471.532x |
| A5 / (10240, 8) / mixed | 2320.35 | 18.64 | 124.482x |
| A5 / (81920, 1) / uniform | 2191.30 | 15.42 | 142.129x |
| A5 / (81920, 1) / hot | 13765.25 | 21.73 | 633.501x |
| A5 / (81920, 1) / mixed | 2331.07 | 14.85 | 156.941x |

All observed regressions are below (INT64 IDs, mask stride 1):

| Platform | Shape | Pattern | Main (us) | Two-stage (us) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| A3 | (1, 1, 1) | uniform | 43.14 | 67.71 | 0.637x |
| A3 | (1, 1, 1) | hot | 42.60 | 67.69 | 0.629x |
| A3 | (1, 1, 1) | mixed | 41.06 | 67.80 | 0.606x |
| A3 | (2, 1, 1) | uniform | 43.35 | 67.76 | 0.640x |
| A3 | (2, 1, 1) | hot | 43.51 | 67.82 | 0.642x |
| A3 | (2, 1, 1) | mixed | 43.82 | 67.79 | 0.646x |
| A5 | (1, 1, 1) | hot | 17.59 | 17.89 | 0.983x |
| A5 | (1, 8, 128) | hot | 15.27 | 15.62 | 0.978x |
| A5 | (1, 8, 128) | mixed | 13.27 | 15.54 | 0.854x |
| A5 | (2, 1, 1) | uniform | 15.03 | 21.57 | 0.697x |

A3 regressions are confined to one or two routes and one expert in this
matrix. A5 regressions also occur at tiny inputs. The two launches and
histogram/scratch work can exceed the scan saving at these sizes. The
implementation does not establish a speedup at every shape.

A separate full msprof capture of uniform `[10240,8]` into 128 bins contains
305 records per stage. Excluding the first 105 eager/warmup invocations leaves
200 steady Graph invocations. Mean `Task Duration(us)` per stage is:

| Platform | Producer (us) | Reducer (us) |
| --- | ---: | ---: |
| A3 | 348.501 | 1.542 |
| A5 | 16.217 | 2.331 |

These are separate task durations, not the complete Graph Event latency;
they are not added to the Graph timing above.

Every point, status, median, observed min/max, ten raw samples per path and
source identity is in the [A3 CSV](eplb_count_graph_a3.csv) and
[A5 CSV](eplb_count_graph_a5.csv). Missing results have empty numeric fields.
Model accuracy, serving performance, startup/serving compilation counts and
repository-wide CI have not been measured by these standalone tests.

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
