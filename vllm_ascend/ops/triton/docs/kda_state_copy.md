# KDA state copy: PR #17301-aligned Triton integration

## Scope and dispatch

The worker selects by device and cache layout without an additional configuration
or environment switch. A5/950 with FP32/BF16, positive `[N,H,V,K]`, dense inner
payload and nonoverlapping cache pages uses the fused plan. A3/non-950 and
unsupported layouts use a masked byte-copy plan prepared during startup.
Triton replaces the native gather/clear and scatter kernels on eligible A5
caches, not chunk math, recurrent decode, GLM KDA or unrelated kernels.

Flow: worker binds cache -> validate and publish plans -> late-register bound
layouts in `kernel_warmup()` -> registry compile-only warmup -> prefill gather ->
unchanged chunk math -> scatter. Both V1 and V2 runners bind plans before warmup.
Preparation reads metadata only; it neither allocates device tensors nor launches
kernels. Invalid layouts abort initialization before publishing a partial worker.
The byte-copy fallback retains Torch clearing and invalid-row masks.

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
- The fallback gathers invalid indices as zero and skips invalid scatter
  destinations, including contiguous unsupported caches.

## Registry warmup and shared launch path

`KDAStateCopyKernel` derives four keys per bound layout: INT32/INT64 indices
multiplied by gather/scatter. Keys retain the actual cache dtype, row count,
page stride, payload size and pointer alignment. Runtime scalars reproduce their
Triton integer specialization, including 64-bit page strides. Metadata and packed
buffers remain aligned and contiguous; gather flags are BOOL, while scatter uses
its index pointer for the unused flag argument. The block size stays 8192.
Selected count appears only in the two-dimensional grid, so all admitted counts
from zero through `scheduler.max_num_seqs` use the same four variants.

`register_kda_state_copy_warmup()` registers only bound layer plans inside the
shared registry window immediately before `registry.warmup()`. Cache layout and
alignment are unavailable during model construction, which requires this late
registration. Equal layouts are registered once. A non-950 or unsupported-dtype
plan instead registers one aligned INT64 `BatchMemcpyKernel` key.

Warmup uses `TritonWarmupTensor` and native `kernel.warmup(grid=(1,))`: it compiles
without allocating scratch tensors or reading/writing the live cache. Serving
uses the same `@kernel_launcher` wrapper and normal Triton cache lookup, avoiding
manual compiled-launch argument lists and their version-dependent constexpr ABI.
The shared registry controls parallel compilation and `enable_jit_warmup`; when
warmup is disabled, ordinary lazy compilation remains available.

Plans retain cache metadata and layer names, with no request tensors, live
pointers, stream handles, private compiled-object tables or environment snapshots.
An unsealed plan, changed layout or excess selected count is rejected before
launch. Compiler/runtime environment changes require a worker restart, as for
other Triton kernels. There is no per-copy environment scan. Current streams and
devices are resolved by the normal launcher on each invocation.

## Dynamo and FakeTensor boundary

Eager plan calls use the shared Triton launcher. During Dynamo tracing, plans
bound by worker initialization emit `vllm::kda_state_gather` and
`vllm::kda_state_scatter`. Their Fake implementations validate pointer-free
metadata and model output allocation / cache mutation. Actual execution resolves
the sealed plan from the current forward context's `no_compile_layers`, as other
Ascend custom ops do. Standalone `prepare()` plans without a worker binding are eager/graph
APIs, not exportable layer-context handles.

These are not ABI replacements for `_C_ascend::kda_state_copy`; they preserve the
prefill copy semantics through a worker-owned lifecycle. Tests cover FakeTensor,
`torch.compile(backend="eager", fullgraph=True, dynamic=True)`, and NPU graph
replay of the traced callable, including FP16 strided fallback. They do not prove
full-model compilation or every compilation backend.

## Validation

`tests/ut/model_executor/warmup/test_jit_warmup_kda_state_copy.py` covers the
key-space golden set, aligned/misaligned caches, 64-bit strides, metadata-only
preparation, registration windows, layout deduplication, fallback registration,
compile-only warmup and the runtime two-dimensional grid.

NPU copy tests cover exact FP32/BF16 values, INT32/INT64 indices, gaps/offsets,
invalid rows, flags, empty selections, more than 4 GiB address strides, dynamic
counts, current streams, graph replay, and the FP16 byte-copy fallback. Standalone
tests explicitly run registry warmup before forbidding compiler APIs. Normal JIT
cache lookup remains allowed; any new compilation during serving fails the test.
Optional native-reference comparisons require a separately built PR #17301 op.

```bash
python -m pytest --confcutdir=tests/ut/model_executor/warmup \
  tests/ut/model_executor/warmup/test_jit_warmup_kda_state_copy.py
python -m pytest --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops/triton \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_state_copy_integration.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_state_copy_triton.py
```

## Performance and acceptance boundaries

Compare startup cold/heat cache and runtime eager/graph timings separately, with
identical layouts, allocation boundaries and compiler versions. Include output
allocation and metadata normalization when measuring production calls. Operator
tests establish copy and tracing behavior; they do not establish full-model
generation quality, distributed serving, sleep/wake behavior or throughput.
