# EPLB all-gather route counting

## Description

- **Function:** Count valid physical routes owned by local experts. This shared
  EPLB utility is called through `distributed/eplb/tensor_ops.py::count_allgather_tokens`
  from the all-gather recording path in `eplb_state.py`.
- **Formula:** `out[e] = sum(mask[t] and ids[t,j] == start + e)`.
  Duplicate routes count separately; masked and nonlocal routes count zero.
- **Algorithm flow:** The first kernel balances fixed 512-route tiles across
  `min(vector_cores, ceil(T*K/512))` programs. Each accumulates a private INT64
  histogram and writes one complete row of `[programs, padded_bins]` scratch.
  The second kernel uses one program to read complete adjacent scratch rows,
  reducing the producer dimension in row tiles sized from an element budget.
  It accumulates an INT64 bin vector and writes the final counts once.
  No global atomic updates are used. Empty T/K returns zeros.
- **Supported modes:** Ascend backend with the worker-selected NPU. Tested
  products are A3 (`Ascend910_9362`) and A5 (`Ascend950DT_9582`). A2 and other
  product variants are unvalidated.

## Parameters

All parameters are required. Public implementation wrapper:
`vllm_ascend.ops.triton.eplb_count.count_allgather_tokens_triton`.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `topk_ids` | Input | Physical expert IDs | INT32/INT64 | Contiguous `[T,K]` |
| `token_mask` | Input | Whether each token contributes | BOOL | `[T]`, stride supported |
| `local_expert_start` | Attribute | First locally owned expert | Python int | Nonnegative scalar |
| `local_expert_count` | Attribute | Local expert count | Python int | Positive scalar |
| return value | Output | Fresh local route counts | INT64 | `[local_expert_count]` |

## Constraints

IDs and mask must be on the same NPU with matching T. Noncontiguous IDs are
rejected without an implicit copy. IDs use direct `ids_ptr + offsets` addressing;
mask strides are supported. Runtime route count and expert bounds use the
backend's INT32 scalar domain; IDs and accumulated counts use INT64. Large IDs
outside the local range are ignored. This is counting, not weighted scatter-add.

`padded_bins = max(2, next_power_of_2(local_expert_count))`; the minimum two
keeps the loop-carried accumulator vector-shaped for the backend. Tile size 512
is shared across devices. The producer grid uses the shared capability helper.
The reducer has an 8192-element tile budget, equivalent to 64 KiB of INT64
partial counts. It chooses
`rows_per_tile = min(programs, max(1, 8192 // padded_bins))` and pads this row
count to a power of two. It loads complete rows, reduces axis 0 and accumulates
the result across the loop. Rows beyond the tile or producer count are masked.
For the tested bin range, the shaped load fits the element budget; the bin
accumulator and compiler temporaries require additional UB.

| Padded bins | A3 rows per tile / iterations (40 producers) | A5 rows per tile / iterations (64 producers) |
| ---: | --- | --- |
| 128 | 40 / 1 | 64 / 1 |
| 256 | 32 / 2 | 32 / 2 |
| 512 | 16 / 3 | 16 / 4 |

Scratch takes `8 * programs * padded_bins` bytes: 40 KiB for 40 programs and
128 bins, or 64 KiB for 64 programs. These are logical allocation sizes, not
physical UB measurements. This validation covers up to 512 local experts.

Scratch and nonempty output are fresh allocations, fully overwritten each call.
There is no separate output-clear kernel. The two launches use the same current
stream; Graph replay must preserve their order. Inputs are not mutated and no
tensor values are read on the host. All-gather, record-enable state and persistent
load accumulation remain with the existing caller.

## Origin and Differences

- **Origin:** The shared count contract in upstream main `3ae9de590`,
  unchanged in the merged main `8cd3c3762`,
  `vllm_ascend/distributed/eplb/tensor_ops.py`.
- **Differences:** Main constructs local IDs, eligibility and clamped indices,
  zeros the output and invokes Torch `scatter_add_`. The final NPU implementation
  combines eligibility with local histogram construction, then reduces partial
  counts. CPU inputs retain main's Torch path. NPU IDs must now be contiguous;
  token masks may still be strided.

### Compilation and integration

Production callers use the public wrapper. Producer variants depend on pointer
dtype/alignment, TOPK, mask stride, padded bins, tile and target/compiler options.
Runtime route count and expert start/count use `do_not_specialize`. Reducer
variants depend on producer count, padded bins, row tile and valid expert count,
controlling loop bounds, tensor dimensions and output masking. Changing only
contents does not create a new variant.

Both launches request `compile_mode="simd"`. A5's histogram lowering can still
produce `parallel_mode="mix_simd_simt"`; the option does not guarantee a purely
SIMD binary. Warmup integration: N/A; no new registry or runtime configuration
is introduced. Model startup/serving compilation counts remain unmeasured.

## Test Cases

Reference: main's Torch formula on CPU. INT64 results require exact equality
(`rtol=0`, `atol=0`). Coverage includes both ID dtypes, empty T/K, 511/512/513
route boundaries, 40/64-producer grid transitions, up to 1,048,576 routes,
512 experts, 15/16/17 partial-row reduction boundaries, non-power-of-two bins,
large/invalid IDs, duplicated routes,
disabled/strided masks, noncontiguous-ID rejection and mutable Graph replay.
The large mutable Graph case captures 20 calls and repeatedly changes IDs/masks.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb_count.py
```

The expanded shape set is synthetic interface/boundary coverage. 81920 routes
and 128 bins preserve the task's original workload; added points are not claimed
as model-captured shapes. Model accuracy, serving performance and compilation
counts remain unmeasured.

### Performance methodology

Compare main `3ae9de590` directly against the final two-stage implementation.
Both receive identical inputs and allocate fresh output each call. Full-wrapper
Graph timing includes main's zeroing and the candidate's reduction. Each graph
contains 20 calls; five warmup replays precede ten alternating device Event
samples, divided by 20. Values are medians in microseconds. CPU-oracle checks run
before capture and after replay. Failures have no numeric speedup. No eager
measurement is included.

There are 141 points/device: 43 shapes x three distributions, three extra
INT32 shapes x three distributions, and one strided-mask shape x three
distributions. Uniform cycles through local experts; hot uses one expert;
mixed includes masked tokens and nonlocal routes. Local start is 7.

### Results

Measured on 2026-11-17 against main `3ae9de590`, using the final implementation
at `fac549668`, which includes the adaptive row-tiled reducer and main
`8cd3c3762`. The pinned baseline file is identical in main `3ae9de590` and
`8cd3c3762`. All source hashes and results below refer to this final reducer.

| Platform | Exact regression tests | Main vs two-stage Graph exact checks | Faster points | Observed speedup range |
| --- | ---: | ---: | ---: | ---: |
| A3 | 453/453 passed | 141/141 passed | 115/141 | 0.371x–1.953x |
| A5 | 453/453 passed | 141/141 passed | Shared-target observation only | Not accepted as controlled performance |

Both devices use CANN 9.2.1, Python 3.12.16, Torch 2.10.0+cpu,
Torch-NPU 2.10.0.post7, Triton 3.6.0 and Triton-Ascend 3.6.0.post2.
Both `bishengir-compile` binaries have SHA256
`69b9402f711ccdcd21c6f951b5e211bb2abd397f1f5ff06dd4dc7a88c6f974bc`
(AscendNPU-IR `f94ebe365e88`, LLVM 19.1.7).

| Device | Architecture | Driver | Vector cores |
| --- | --- | --- | ---: |
| A3 Ascend910_9362 | NpuArch2201 | 26.2.rc1.b081 | 40 |
| A5 Ascend950DT_9582 | NpuArch3510 | 26.2.rc1.b088 | 64 |

**A5 timing qualification:** another VLLM process was present on the same
physical device (141.61.45.109, card 1). That process started at 16:35:16 +0800;
the Graph matrix ran from 16:38:20 to 16:43:58 +0800 on 2026-11-17. The exact
checks passed, but these timings are shared-target observations and are not
accepted as controlled performance results. The A5 CSV retains all observed
values with `timing_qualification=shared_target_load_observation_only`.
Available replacement A5 devices were occupied at their pre-launch rechecks;
a run without another workload on the target remains pending.

A3 used a healthy target with no other target-device process observed in the
ownership checks; other host devices had active jobs. It has 26 regressions,
including `(10240,8,512)` uniform: 620.32 us on main versus 1001.70 us here
(0.619x). The worst measured ratio is 0.371x at `(1088,8,512)`. The implementation
does not establish a universal speedup. Every observed range and raw sample
is retained in the CSV; faster-point counts refer to measured medians.

The following table uses INT64 IDs, contiguous masks and uniform local routes.
Shapes are `(T, K, local_expert_count)`. Latencies are microseconds per call.
All A5 columns are shared-target observations as qualified above.

| Shape | A3 main | A3 two-stage | A3 speedup | A5 main | A5 two-stage | A5 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (1, 1, 1) | 43.40 | 67.91 | 0.639x | 10.39 | 26.72 | 0.389x |
| (1, 8, 128) | 102.71 | 70.84 | 1.450x | 16.28 | 15.65 | 1.040x |
| (32, 2, 8) | 90.37 | 70.21 | 1.287x | 33.11 | 16.50 | 2.006x |
| (64, 4, 32) | 104.61 | 78.80 | 1.327x | 46.32 | 4.91 | 9.432x |
| (65, 8, 17) | 107.92 | 90.70 | 1.190x | 138.90 | 11.84 | 11.730x |
| (128, 3, 64) | 107.64 | 85.42 | 1.260x | 38.12 | 6.42 | 5.934x |
| (256, 8, 64) | 129.07 | 95.50 | 1.352x | 132.60 | 11.63 | 11.397x |
| (511, 1, 128) | 116.40 | 87.78 | 1.326x | 31.38 | 22.65 | 1.385x |
| (512, 1, 128) | 115.28 | 87.83 | 1.312x | 30.52 | 2.63 | 11.595x |
| (513, 1, 129) | 115.12 | 100.11 | 1.150x | 31.29 | 16.35 | 1.913x |
| (512, 8, 128) | 144.54 | 114.02 | 1.268x | 142.07 | 7.55 | 18.811x |
| (2048, 8, 128) | 316.19 | 184.29 | 1.716x | 451.04 | 16.21 | 27.827x |
| (10240, 1, 128) | 183.32 | 179.97 | 1.019x | 286.50 | 17.18 | 16.680x |
| (10240, 4, 128) | 505.75 | 367.34 | 1.377x | 1112.89 | 21.15 | 52.628x |
| (10240, 8, 32) | 586.71 | 398.43 | 1.473x | 7477.25 | 27.22 | 274.696x |
| (10240, 8, 64) | 597.64 | 447.08 | 1.337x | 4350.55 | 18.03 | 241.336x |
| (10240, 8, 128) | 614.63 | 545.29 | 1.127x | 2180.00 | 18.65 | 116.862x |
| (10240, 8, 256) | 620.87 | 770.41 | 0.806x | 1168.81 | 22.94 | 50.955x |
| (10240, 16, 128) | 1422.07 | 902.58 | 1.576x | 4306.45 | 38.25 | 112.593x |
| (32768, 8, 128) | 2165.56 | 1347.46 | 1.607x | 6862.35 | 46.48 | 147.646x |
| (2, 1, 1) | 41.58 | 67.86 | 0.613x | 10.54 | 2.81 | 3.753x |
| (16, 2, 16) | 92.64 | 70.14 | 1.321x | 30.78 | 6.89 | 4.464x |
| (127, 8, 127) | 121.66 | 97.95 | 1.242x | 41.59 | 7.16 | 5.809x |
| (129, 8, 129) | 120.64 | 116.45 | 1.036x | 40.90 | 16.35 | 2.502x |
| (1024, 2, 128) | 130.84 | 102.54 | 1.276x | 75.93 | 7.98 | 9.512x |
| (1024, 8, 128) | 209.19 | 137.54 | 1.521x | 236.84 | 17.57 | 13.479x |
| (2559, 8, 128) | 271.81 | 277.88 | 0.978x | 556.60 | 17.54 | 31.728x |
| (2560, 8, 128) | 277.54 | 277.92 | 0.999x | 572.26 | 22.00 | 26.017x |
| (2561, 8, 128) | 268.23 | 364.17 | 0.737x | 562.70 | 9.80 | 57.421x |
| (4095, 8, 128) | 394.68 | 366.38 | 1.077x | 893.91 | 21.97 | 40.686x |
| (4096, 8, 128) | 446.48 | 366.33 | 1.219x | 889.06 | 9.78 | 90.917x |
| (4097, 8, 128) | 455.98 | 366.38 | 1.245x | 884.04 | 21.97 | 40.229x |
| (8192, 8, 128) | 695.05 | 543.86 | 1.278x | 1732.79 | 14.37 | 120.554x |
| (16384, 8, 128) | 1178.88 | 812.08 | 1.452x | 3489.18 | 23.57 | 148.013x |
| (65536, 8, 128) | 4270.75 | 2506.12 | 1.704x | 13757.46 | 76.51 | 179.809x |
| (131072, 8, 128) | 6310.93 | 4822.89 | 1.309x | 28413.73 | 147.06 | 193.208x |
| (10240, 2, 128) | 279.58 | 278.08 | 1.005x | 556.52 | 19.46 | 28.604x |
| (10240, 32, 128) | 2687.91 | 1614.02 | 1.665x | 8645.37 | 49.63 | 174.206x |
| (10240, 8, 16) | 583.93 | 374.73 | 1.558x | 14040.62 | 28.12 | 499.318x |
| (10240, 8, 512) | 620.32 | 1001.70 | 0.619x | 839.30 | 29.66 | 28.298x |
| (960, 8, 512) | 175.57 | 281.39 | 0.624x | 121.56 | 10.98 | 11.075x |
| (1024, 8, 512) | 216.82 | 279.27 | 0.776x | 125.35 | 10.92 | 11.476x |
| (1088, 8, 512) | 187.90 | 506.22 | 0.371x | 130.04 | 14.44 | 9.004x |

For the original 81920-route/128-bin workload:

| Device / distribution | Main (us) | Two-stage (us) | Observed ratio |
| --- | ---: | ---: | ---: |
| A3 / uniform | 614.63 | 545.29 | 1.127x |
| A3 / hot | 715.32 | 523.94 | 1.365x |
| A3 / mixed | 616.48 | 520.93 | 1.183x |
| A5 / uniform | 2180.00 | 18.65 | 116.862x |
| A5 / hot | 13460.52 | 30.17 | 446.088x |
| A5 / mixed | 2308.05 | 26.14 | 88.295x |

All 141 points/device, including regressions, distributions, dtype and
mask-stride controls, are in the [A3 CSV](eplb_count_graph_a3.csv) and
[A5 CSV](eplb_count_graph_a5.csv). Each row includes status, median, observed
range, ten raw samples per implementation, baseline/candidate identities and
kernel SHA256. Failed points, if any, retain errors and empty numeric fields.

An earlier A5 multi-Graph correctness failure remains without an established
root cause. It did not recur in this matrix or the current mutable-Graph
regressions. Model accuracy, serving performance and startup/serving compilation
counts are not measured by these operator tests.

### Reproduce the Graph matrix

Use the same installed dependencies on the selected NPU. From the repository
root, extract the pinned baseline and run the committed benchmark:

```bash
mkdir -p /tmp/eplb-main-baseline
git show 3ae9de5900e181817ad86a0fa936c379a3789c0f:vllm_ascend/distributed/eplb/tensor_ops.py > /tmp/eplb-main-baseline/baseline_tensor_ops.py
PYTHONPATH="$PWD:/tmp/eplb-main-baseline:$PYTHONPATH" python benchmarks/eplb_count_graph.py
```

The runner saves every point and raw timing sample to `results/expanded-graph.json`,
including errors with null timings and speedups, and records imported source
paths and SHA256 hashes in `results/import-identity.json`. It performs Graph
measurements only. Use separate output directories for different devices/runs.
