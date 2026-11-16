# EPLB all-gather route counting

`count_allgather_tokens_triton(ids, mask, start, count)` implements the NPU path
of `distributed/eplb/tensor_ops.py::count_allgather_tokens`. CPU inputs retain
the existing Torch reference. Inputs are INT32/INT64 physical expert IDs [T,K],
a BOOL token mask [T], and the nonnegative local start and positive count C.
Output is a fresh INT64 [C] tensor: `out[e] = sum(mask[t] and ids[t,j] == start+e)`.
Repeated routes count separately; masked, negative and nonlocal routes count zero.
Strided inputs and empty T/K are supported.

The output is zero-initialized. Fixed 512-route tiles are balanced across a grid
bounded by the Vector Core count from the shared capability helper. Each program
accumulates a private INT64 histogram across all its tiles, then atomically merges
its nonzero bins into the output once. Histogram length is the next power of two
of C, with a minimum of two: the A5 backend failed to lower a one-element
loop-carried tensor, including an attempted sum reduction. Padding resolves that
case without changing the 128-bin main case. The caller selects the current NPU,
following the normal worker convention.

No host reads of tensor values or global intermediate index/mask tensors are
introduced. The 128-bin logical accumulator is 1 KiB. Saved compiler IR confirms
the loop-carried histogram and atomic store outside the loop; final physical UB
placement and compiler-internal workspace have not been established.

The caller still owns gathering/padding, record-enable state and persistent
load accumulation. This is the existing 0/1 route-count contract, not a general
weighted scatter API.

## Validation

The current launch explicitly sets `compile_mode="simd"` to select pure SIMD
on Ascend 950. A2/A3 already use SIMD under the default hybrid mode. See the
[compiler options reference](https://triton-ascend.readthedocs.io/en/v3.6.0/environment_variable_and_compiler_options_reference.html).
This compiler-option change is pending device validation: the available A5
cards were mapped by other running containers at the latest resource check.
The measurements and compiler evidence below belong to commit `980f446c2`,
before this option was added, and do not validate the forced-SIMD candidate.

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
