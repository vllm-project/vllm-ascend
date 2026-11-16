# EPLB all-gather route counting

`count_allgather_tokens_triton(ids, mask, start, count)` replaces the NPU
implementation of `distributed/eplb/tensor_ops.py::count_allgather_tokens`.
CPU inputs retain the existing Torch reference.

Inputs are physical expert IDs `[T,K]` (INT32 or INT64), an aligned BOOL token
mask `[T]`, and the nonnegative local expert start and positive local count C.
Strided inputs are supported. Output is a fresh INT64 `[C]` tensor, defined by
`out[e] = sum(mask[t] and ids[t,j] == start+e for t,j)`.
Duplicate routes each count once; padded, masked, negative and nonlocal routes
contribute zero. Empty T or K returns zeros.

Torch initializes the output on the input device. A Triton kernel then visits
512-assignment tiles and distributes them evenly across a grid bounded by the
Vector Core count from the shared device-property helper. Each program retains
an INT64 expert histogram across its assigned tiles, then merges nonzero bins
with one vectorized atomic add. Histogram length is the next power of two of C;
the 128-expert case has 128 private counters (1 KiB of logical state).
Invalid lanes are excluded from each tile histogram. IDs retain their width
before localization and eligibility checks. A single expert uses a sum reduction
because the tested A5 compiler fails to convert a one-bin histogram.
No tensor values are read on the host. Intermediate global index, clamped-index
and eligibility INT64 tensors are eliminated. The caller must select the current
NPU, matching the worker convention used by the shared capability helper.

The kernel follows the tiled scheduling idea used by the V2 penalty bincount,
but does not perform prompt-flag generation or bitmap packing. It implements
the specific 0/1 route-count scatter, not a general weighted scatter API.

The caller still owns mask gathering and padding alignment, record-enable state,
and persistent load accumulation. This change does not route additional hardware
or communication modes into the all-gather EPLB path.

The current local-histogram revision is a draft awaiting A3 and A5 acceptance.
The initial A5 trial failed on the one-bin histogram; the 128-bin trial reached
binary loading but failed to map a temporary shared library from `/tmp`.
The single-bin reduction and project-local temporary directory are prepared,
but the rerun was interrupted by SSH connectivity failure. A3 initialization
was blocked by thread creation failure in the selected isolated containers.
There are no completed correctness or performance results for this revision.
Physical UB placement has not yet been confirmed from compiler evidence.

The following results belong to the previous direct-atomic implementation
at commit `28ccdd63f5ebbac332ed10e5c3b003fb4501b28d`, not the histogram revision.
A5 validation on Ascend950DT_9582 (CANN 9.2.1, Torch-NPU 2.10.0.post7,
Triton-Ascend 3.6.0.post2) passed all 50 integer-exact differential cases and
the mutable-input graph replay test. A3 and model-level integration remain
unvalidated. INT64 atomic compatibility on other backends is not inferred
from the existing INT32 penalty bincount implementation.

Synthetic tests exercise the actual `count_allgather_tokens` dispatch, tails,
strides, local ranges, repeated IDs, masks, empty inputs and mutable graph replay:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_eplb_count.py
```

For synthetic INT64 IDs `[10240,8]`, output `[128]` and local start 7, complete
function Graph timing gave the following medians. Each Graph contains 20 calls;
after 5 warmup replays, 10 samples alternate baseline/candidate execution order.
NPU Event replay duration is divided by 20. Both implementations are checked
against the CPU reference before and after measurement.

| Distribution | Torch count (ms) | Triton count (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Uniform local routes, all tokens valid | 2.2001 | 2.1336 | 1.031x |
| All routes target one local expert | 13.9165 | 13.4789 | 1.032x |
| Mixed local/nonlocal routes and masked tokens | 2.3427 | 1.4117 | 1.660x |

Uniform IDs cycle through `start + [0,127]`. Hotspot IDs all equal `start`.
Mixed IDs cycle through `[start-2,start+129]`, with every third token disabled.
These synthetic distributions do not establish production-model speedups.

Separate full msprof captures for the uniform case show 9 compute kernels per
original count call versus 2 for the candidate (zeroing and counting). After
excluding 5 warmup calls, 20 samples give mean scatter-kernel durations of
2140.433 us for Torch and 2145.109 us for Triton. The complete function improves
slightly while the counting kernels take similar time; hotspot contention
remains expensive. These eager profiler durations are distinct from Graph
Event timing.
