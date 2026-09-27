# Explicit-lifecycle KDA state copy

## Scope

This opt-in Triton module copies dense KDA payloads between packed states and
first-axis-strided cache pages. It is **not wired into production Kimi prefill**,
does not replace an existing native operator, and does not change decode.
It can be reviewed independently of the native implementation in PR #17301.
Production startup integration, dispatch policy and model-level validation are
separate requirements before enabling it automatically.

## Lifecycle

Flow: disposable startup samples -> prepare every admitted signature -> seal ->
copy with prepared launchers; reject an unknown signature without JIT fallback.

1. Import `prepare_kda_states_triton`, `seal_kda_states_triton` and
   `copy_kda_states_triton` from `vllm_ascend.ops.triton.kda_state_copy`.
2. Call `prepare_kda_states_triton` with representative disposable tensors for
   every admitted signature, using the same keyword arguments as serving.
   Preparation **executes** gather/scatter and can overwrite its output/cache.
3. Call `seal_kda_states_triton()` before serving or capturing a serving graph.
   This synchronizes prepared devices and irreversibly seals the process-local
   registry. Repeated seal calls are idempotent.
4. Use `copy_kda_states_triton` during serving. An unprepared signature raises
   `RuntimeError`; do not catch it and silently switch to an on-demand JIT path.

The default block size is 1024 elements. The acceptance runs explicitly used
8192. Prepare and serve with the same block size; neither value is asserted to
be optimal across shapes or hardware.

## Input and signature contract

- State: same-device FP32/BF16 `[cache_rows, H, V, K]`, dense inner payload,
  positive dimensions and first-axis stride at least the payload size.
- Packed: contiguous `[selected_rows, H, V, K]`, same dtype/device as state.
- Indices: one-dimensional INT32/INT64 device tensor; strided vectors are allowed.
- Flags: optional initial-state flags, normalized to a one-dimensional BOOL
  tensor on the state device. Empty selections still undergo validation.
- Gather copies valid rows; invalid indices and false flags produce zeros.
  Scatter skips invalid indices and ignores initial-state flag values.
- Legal scatter destinations must be unique. State and packed storage must not
  overlap. These are caller preconditions, not host-side uniqueness/alias scans.
- Page gaps, storage offsets and addresses beyond 4 GiB are supported by masked,
  64-bit element address arithmetic. No full-cache copy is performed.

Signatures include device, dtype, index dtype, effective pointer alignment modulo
16, scalar layout metadata, selected count/grid, direction, flag presence and
block size. Tensor identity and index/flag **values** are not cached. Do not
prepare with a compact clone when serving will use gapped pages or a different
storage offset/alignment. Changing the H/V/K decomposition may share a signature
only after the normal dense-inner-layout validation succeeds.

The registry allows up to 128 entries, with no eviction. Exceeding the budget
fails during preparation. A new process must prepare and seal again. Runtime
configuration drift and JIT pre-run hooks are rejected. This module pins existing
compiler-related environment values; it does not introduce a configuration switch.
The process-local mutable registry and its ownership/lifecycle need architectural
review before production integration.

## Reproducing the tests

CPU host-lifecycle tests use isolated Torch/Triton stubs. They require pytest but
not an NPU and do not prove device numerical correctness.

```bash
pytest --confcutdir=tests/ut/ops tests/ut/ops/test_kda_state_copy_lifecycle.py -q
```

The device tests require a working Ascend Torch/Triton/vLLM environment. Run the
file in its own process, with no concurrent benchmark. The greater-than-4-GiB
case requires sufficient free device memory and is intentionally not skipped.

```bash
pytest --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kda_state_copy_triton.py -v
```

The NPU suite preserves the 17 parameterized cases and exact assertions from
PR #17301, commit `5bcbad36fdc30aef539abf0ccc99ef959d913104`, while adapting the
operator entry. Its fixture executes the same cases once to prepare signatures,
seals the registry and prohibits compiler API entry for the formal pytest run.
The preparation executions are not additional passing pytest cases. The test
loads the module from the checkout so an older installed file cannot shadow it.

## Evidence and limitations

Before repository packaging, the standalone strict implementation passed 132
acceptance records (including preparation, repeated checks and positive controls,
not 132 unique contracts). A separate original-PR pytest rerun passed 21 reference
CPU tests, 17 native NPU tests and 17 strict Triton NPU tests. Those reference CPU
tests do not validate this module's production dispatch. The native tests used an
existing binary, not an attested clean build of the native PR.

Recorded standalone environment: Ascend950DT, Torch 2.10.0+cpu,
torch_npu 2.10.0.post4 and Triton module version 3.2.0. Packaging verification
identified the installed distribution as triton-ascend 3.2.2+dev20260729205041.
The exact released dependency set and other hardware require separate validation.
Final-checkout test results must be attached to the PR; historical results are not CI.

No strict-wrapper performance claim is made. Historical cached-launcher A
benchmarks are not measurements of this wrapper and are deliberately excluded.
Numerical tests use zero tolerance; they do not establish full-model accuracy.

## Packaging validation (2026-09-27)

The packaged module's executable AST matches the previously accepted strict
implementation after excluding imports and docstrings. Only import order,
formatting and documentation were changed. The original device-test function
ASTs also match after replacing the operator entry with the local adapter.

- Packaged host-lifecycle suite: 23 passed on CPU.
- Packaged device suite, after final Python formatting: 17 passed, no failures
  or skips. It emitted 14 existing Torch JIT deprecation warnings. The formal
  phase forbids compiler API entry and retains zero-tolerance assertions.
- Ruff 0.14.0 lint/format, forbidden-import, boolean-context-manager,
  Python-package-init and diff-whitespace checks passed for the checked scope.
- Full `format.sh ci` was attempted but blocked by missing `pre-commit` in the
  local environment. This is not a full repository CI pass.
- Tests load the packaged source against the existing NPU runtime; they do not
  prove a clean install or whole-model compatibility of the complete new base.

No throughput or latency measurements were collected during packaging.

## Boundary conditions and further work

- Negative indices do not wrap around as Legacy advanced indexing does.
  Mixed state/packed dtypes are rejected rather than automatically converted.
- Graph replay requires fixed pointer lifetimes; the tested dynamic changes are
  device indices and BOOL flag values, not replacement of captured tensors.
- Conflicting concurrent writes and repeated valid scatter targets are undefined.
  The host registry lock does not serialize asynchronous device writes.
- Sealing only constrains this entry point in a fixed environment. It is not a
  process-wide compilation sandbox or protection from Python monkey-patching.
- Production signature enumeration, initialization, dispatch/fallback policy,
  optional operator/Meta registration and end-to-end model testing remain open.
- Before changing host dispatch for speed, measure the exact submitted version
  against same-batch baselines and preserve eager regressions in the report.
