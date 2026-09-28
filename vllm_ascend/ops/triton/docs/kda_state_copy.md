# KDA state copy: PR #17301-aligned Triton integration

## Scope and dispatch

The default `kda_state_copy_backend="auto"` follows the PR #17301 capability
policy: A5/950, FP32/BF16, positive `[N,H,V,K]`, dense inner payload and
nonoverlapping cache pages. Triton replaces the native gather/clear and scatter
kernels, not chunk math, recurrent decode, GLM KDA or unrelated Triton kernels.

`AscendConfig` also accepts `triton` to explicitly require a prepared fused plan
and `torch` for the prior diagnostic baseline. No option is required for the
automatic production path. The legacy `torch` override is not the automatic
strided fallback and is not included in the zero-serving-JIT guarantee.

Flow: worker binds cache -> choose fused / strided fallback / contiguous fallback
-> prepare disposable samples -> seal all plans -> publish -> prefill gather ->
unchanged chunk math -> scatter. Preparation failure aborts startup; it must not
silently downgrade a requested fused path or publish a partially ready worker.

Both V1 and V2 runners invoke initialization after cache binding. An automatic
layer cannot serve until initialization has completed. Unsupported contiguous
caches retain indexing and Torch clearing. Unsupported noncontiguous caches use
the PR's byte-pointer copy algorithm with a startup-precompiled batch-memcpy
launcher. The fallback uses Torch clearing to avoid a second lazy Triton kernel.
Invalid inner layouts remain rejected by the strided fallback, as in the PR.

## Numerical contract

- Fused gather produces zero for invalid indices and false initial-state flags.
  Missing flags mean all valid rows are gathered. Repeated gather indices work.
- Fused/byte-copy scatter skips invalid indices; valid destinations must be
  unique. Initial-state flags do not suppress final writes.
- Cache page gaps and layer storage offsets are not copied or overwritten.
  Address arithmetic widens to INT64 before multiplication, including INT32
  index inputs and pages more than 4 GiB apart.
- Packed outputs have the cache dtype. Final states are converted to that dtype
  and contiguous storage before scatter, as in PR #17301.
- Empty selections validate metadata and do not launch a copy kernel.
- Cache/packed storage must not overlap. Concurrent writers require upstream
  coordination. These are caller preconditions, not synchronizing host scans.
- The unsupported contiguous indexing fallback retains the PR's original
  indexing semantics; its invalid-index behavior is not the fused contract.

## Startup preparation and no serving recompilation

`KDAStateCopyPlan` prepares exactly four compiled variants per bound cache layout:
INT32/INT64 indices multiplied by gather/scatter. All variants use fixed cache
metadata and block size 8192. Gather flags are aligned contiguous BOOL. Index
values and selected counts are runtime data; launch grid is rebound without a
JIT call. `0 <= selected <= scheduler.max_num_seqs` does not expand the registry.

Preparation uses one disposable payload row and small metadata tensors. The
actual row count, stride and alignment are scalar/layout information only; all
kernel pointers refer to scratch allocations. Index zero ensures the scratch
never needs to span the live cache. Startup synchronizes before releasing it.
The strided fallback separately prepares one byte-copy variant with scratch
source/destination pointers, not live cache pointers.

Serving invokes only stored compiled objects. An unsealed plan, changed cache
layout, excess selected count, or unexpected JIT hook is rejected before launch.
No request can add a variant or evict one. Different worker layouts require
reinitialization; equal layouts may share plans. Plans retain no request tensors,
stream handles or live data pointers.

Aligned contiguous INT32 inputs remain INT32: no per-call INT64 conversion.
Strided/misaligned metadata is normalized only when needed, preserving dtype.
This bounded normalization avoids treating arbitrary strides/alignment classes
as new compiler signatures. Pointer addresses and metadata values can change.
The current stream is resolved by the compiled launcher for each invocation.

### Compiler-environment contract

Compiler environment/version snapshots are compared during `prepare`/`seal`,
not scanned on every gather/scatter. Changing compiler environment, backend
binaries or loaded implementation after startup is unsupported: restart and
prepare again. This is a deliberate change from the previous strict wrapper's
per-call drift detection. It does **not** make a stored compiled object capable
of JIT compilation, nor does it remove tensor-layout validation.

Do not reintroduce per-copy `os.environ` enumeration, or convert every INT32
index tensor to INT64 simply to minimize startup variant count. Both were
measured regressions in the preceding implementation.

## Dynamo and FakeTensor boundary

Eager plan calls keep the direct compiled launch. During Dynamo tracing, plans
bound by worker initialization emit `vllm::kda_state_gather` and
`vllm::kda_state_scatter`. Their Fake implementations validate pointer-free
metadata and model output allocation / cache mutation. Actual execution resolves
the sealed plan from the current forward context's `no_compile_layers`, as other
Ascend custom ops do. No extra global plan registry or runtime compilation is
introduced. Standalone `prepare()` plans without a worker binding are eager/graph
APIs, not exportable layer-context handles.

These are not ABI replacements for `_C_ascend::kda_state_copy`; they preserve the
prefill copy semantics through a worker-owned lifecycle. Tests cover FakeTensor,
`torch.compile(backend="eager", fullgraph=True, dynamic=True)`, and NPU graph
replay of the traced callable, including FP16 strided fallback. They do not prove
full-model compilation or every compilation backend.

## Validation

CPU tests cover enum defaults, automatic routing, startup deduplication,
all-or-nothing publication and actual prefill dispatch. NPU integration tests
cover optional exact FP32/BF16 comparisons with a separately built PR17301 native operator,
INT32/INT64, dynamic selected counts, gaps/offsets, invalid indices, graph replay,
real prefill/chunk math, and the FP16 strided byte-copy fallback.

Formal serving tests replace the KDA JIT entry and compiler APIs with functions
that fail immediately. Fallback tests also disable the byte-copy JIT entry.
The suite separately asserts that aligned INT32 inputs retain their identity
and serving never invokes the compiler-environment scanner. Changing selected
counts and graph input values must succeed with a constant variant table.

Run the integration tests before the standalone suite if collecting them in one
process; the latter uses an isolated module and compiler-denying fixture:

```bash
python -m pytest --confcutdir=tests/ut/ops \
  tests/ut/ops/test_kda_state_copy_lifecycle.py \
  tests/ut/ops/test_kda_state_copy_integration.py \
  tests/ut/ops/test_kimi_kda.py
python -m pytest --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kda_state_copy_integration.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kda_state_copy_triton.py
```

## Performance acceptance and remaining boundaries

Use identical inputs and allocation boundaries for native versus Triton
production calls. Include output allocation, necessary metadata work and final
state conversion. Compare old and new implementations in the same process and
rotate order. Record eager Event, synchronized wall time, and graph replay
separately; an Event interval includes host dispatch gaps, not just kernel time.
A historical preallocated operator benchmark is a secondary reference, not a
substitute for production-wrapper timing.

Local tests are not full model-server, TP/PP, concurrency, sleep/wake or generation
quality acceptance. The native comparison reuses the installed extension and is
not a clean Ascend C build. Non-950 automatic fallback is code-routed but requires
additional hardware validation. Full-model compilation/Inductor acceptance is
not established by operator tracing tests. The no-JIT guarantee here covers
prepared state-copy routes, not unrelated model kernels.

The standalone explicit prepare/seal API in `ops/triton/kda_state_copy.py` remains
available for its existing tests; production uses worker-owned plans instead of
its process-global signature registry and per-call environment checks.

### Reproducible paired performance check

The optional native differential tests skip only when the separately built
PR #17301 reference op is absent; all independent correctness tests remain
required. Production does not depend on that op. The benchmark requires it
explicitly and fails early if unavailable:

```bash
python -m tests.e2e.nightly.single_node.ops.singlecard_ops.benchmark_kda_state_copy_triton \
  --output /tmp/kda-round-1.json --seed 41 --trials 100
```

Use a fresh process and a new output file for each seed. The benchmark includes
production allocation, a native predicate ablation, equal preallocated
boundaries, and two labels for the identical native callable as a measurement
control. Every trial shuffles order. Output lifetimes end outside the timed
intervals. It records all raw samples and implementation hashes, denies Triton
compiler entry after warmup, and reports host enqueue, synchronized wall, eager
Event and 100-call graph replay separately. Never interpret eager Event as pure
kernel time or combine different rounds/modes by subtracting their medians.
