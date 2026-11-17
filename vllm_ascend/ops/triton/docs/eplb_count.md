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
  The second kernel sums the program dimension and writes the final counts.
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
The reducer is one program, with its reduction dimension padded to a power of
two. Scratch takes `8 * programs * padded_bins` bytes: 40 KiB for 40 programs and
128 bins, or 64 KiB for 64 programs. These are logical allocation sizes, not
physical UB measurements.

Scratch and nonempty output are fresh allocations, fully overwritten each call.
There is no separate output-clear kernel. The two launches use the same current
stream; Graph replay must preserve their order. Inputs are not mutated and no
tensor values are read on the host. All-gather, record-enable state and persistent
load accumulation remain with the existing caller.

## Origin and Differences

- **Origin:** The shared count contract in upstream main `3ae9de590`,
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
variants depend on producer count, padded reduction size, padded bins and valid
expert count, controlling tensor dimensions and output masking. Changing only
contents does not create a new variant.

Both launches request `compile_mode="simd"`. A5's histogram lowering can still
produce `parallel_mode="mix_simd_simt"`; the option does not guarantee a purely
SIMD binary. Warmup integration: N/A; no new registry or runtime configuration
is introduced. Model startup/serving compilation counts remain unmeasured.

## Test Cases

Reference: main's Torch formula on CPU. INT64 results require exact equality
(`rtol=0`, `atol=0`). Coverage includes both ID dtypes, empty T/K, 511/512/513
route boundaries, non-power-of-two bins, large/invalid IDs, duplicated routes,
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

There are 72 points/device: 20 shapes x three distributions, three extra
INT32 shapes x three distributions, and one strided-mask shape x three
distributions. Uniform cycles through local experts; hot uses one expert;
mixed includes masked tokens and nonlocal routes. Local start is 7.

### Results

Measured on 2026-11-17 against main `3ae9de590`. A3 used the full source
snapshot at `962225ae5` with the expanded test overlay. The production kernel
is unchanged in the expanded validation branch.

| Platform | Exact regression tests | Main vs two-stage Graph points | Faster points |
| --- | ---: | ---: | ---: |
| A3 | 223/223 passed | 72/72 passed | 67/72 |
| A5 | Expanded run pending | Not measured in this matrix | N/A |

A3: Ascend910_9362, NpuArch2201, driver 26.2.rc1.b081, 40 vector cores.
Software: CANN 9.2.1, Python 3.12.16, Torch 2.10.0+cpu,
Torch-NPU 2.10.0.post7, Triton 3.6.0 / Triton-Ascend 3.6.0.post2.
This was a single-device run on a shared host; other devices had active jobs.

A3 speedup ranges from 0.665x to 1.896x. Five points regress; the singleton
case is approximately 0.67x. The two-stage path is not a universal speedup.
The following table uses INT64 IDs, contiguous masks and uniform local routes.
Shapes are `(T, K, local_expert_count)`, and latency is microseconds per call.

| Shape | Main (us) | Two-stage (us) | Main / two-stage |
| --- | ---: | ---: | ---: |
| (1, 1, 1) | 45.07 | 67.75 | 0.665x |
| (1, 8, 128) | 106.09 | 70.86 | 1.497x |
| (32, 2, 8) | 94.96 | 70.08 | 1.355x |
| (64, 4, 32) | 108.70 | 78.79 | 1.380x |
| (65, 8, 17) | 109.09 | 90.75 | 1.202x |
| (128, 3, 64) | 112.92 | 85.49 | 1.321x |
| (256, 8, 64) | 132.83 | 95.40 | 1.392x |
| (511, 1, 128) | 118.64 | 87.71 | 1.353x |
| (512, 1, 128) | 116.75 | 87.74 | 1.331x |
| (513, 1, 129) | 117.49 | 100.17 | 1.173x |
| (512, 8, 128) | 148.44 | 114.02 | 1.302x |
| (2048, 8, 128) | 316.07 | 184.29 | 1.715x |
| (10240, 1, 128) | 188.89 | 180.55 | 1.046x |
| (10240, 4, 128) | 511.43 | 366.84 | 1.394x |
| (10240, 8, 32) | 600.64 | 398.48 | 1.507x |
| (10240, 8, 64) | 604.91 | 446.73 | 1.354x |
| (10240, 8, 128) | 621.29 | 545.34 | 1.139x |
| (10240, 8, 256) | 623.68 | 737.75 | 0.845x |
| (10240, 16, 128) | 1417.02 | 901.53 | 1.572x |
| (32768, 8, 128) | 2177.10 | 1347.22 | 1.616x |

All 72 cases, including regressions, distributions, dtype and mask-stride
controls, are in [the A3 result CSV](eplb_count_graph_a3.csv). Each row includes
the median and observed sample range. Raw samples are written by the benchmark.

A5 expanded validation is blocked by GitHub transfer through the current
proxy's untrusted certificate chain. No A5 main-comparison numbers are inferred
from A3 or from other benchmark protocols. A separate A5 multi-Graph correctness
failure has not been root-caused; passing subsequent checks does not resolve it.

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
