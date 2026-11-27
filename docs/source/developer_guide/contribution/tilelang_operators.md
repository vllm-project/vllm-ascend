# Contributing TileLang operators

See [RFC #23332](https://github.com/vllm-project/vllm-ascend/issues/23332)
for the TileLang integration roadmap.

## Deliverables checklist

- [ ] PR description.
- [ ] Kernel source and a public Python wrapper under `vllm_ascend/ops/tilelang/`.
- [ ] Markdown operator document covering functionality, algorithm, parameters, constraints, origin and test entry points.
- [ ] Operator tests; warmup unit tests, when applicable: `tests/ut/model_executor/warmup/`.
- [ ] Repository-wide deliverables: see the [contribution guide](index.md) and [release note guide](https://github.com/vllm-project/vllm-ascend/blob/main/release_notes/README.md).

## Overall objectives

TileLang operators merged into the project should meet quality requirements in four areas:

1. **Functionality:** Implement the required functionality and preserve compatibility across applicable devices supported by the project.
2. **Accuracy:** Meet operator-level and model-level accuracy criteria.
3. **Performance:** Performance optimization PRs improve over the baseline. Other PRs avoid unjustified performance regressions.
4. **Clean Code:** Keep code readable and maintainable, use reasonable specialization, and avoid undeclared external side effects.

## Detailed requirements

Contributors must provide the following content in the deliverables before merge.

### Functionality

- [ ] The PR description includes:
    - [ ] Functionality, inputs and outputs, production callers, and whether the operator is framework-wide or model-specific.
    - [ ] Motivation, such as model support, a bug fix, performance optimization, or temporary compatibility. Temporary implementations include a replacement and removal plan.
    - [ ] Cross-device compatibility and actual validation results. Device-specific optimization preserves functionality and correctness on other applicable devices through supported dispatch or an existing fallback.
- [ ] The Markdown document describes functionality, algorithm, parameters, constraints, origin and differences, and test entry points.
- [ ] The implementation matches the documented inputs, outputs and behavior, including dispatch conditions for hardware, dtype, shape, strides and configuration.
- [ ] The operator is integrated into the corresponding model or shared operator call path through an existing CustomOp or attention entry point.
- [ ] The integration provides the operator schema, FakeTensor implementation and mutation declarations required by `torch.compile`.

### Accuracy

- [ ] The PR description includes:
    - [ ] Operator accuracy results against a reference implementation.
    - [ ] Model accuracy results against the baseline, recording the dataset revision, prompt template, sampling settings and seed.
- [ ] Test files provide:
    - [ ] Coverage of all declared input categories and relevant boundaries, including shapes, dtypes and layouts.
    - [ ] For framework-wide operators, coverage of the general interface and representative real shapes from applicable models.
    - [ ] For model-specific operators, coverage of real shapes and parameter combinations reachable in the target models under the declared configurations.
    - [ ] Parameterized representative cases and boundaries, avoiding redundant cases.
    - [ ] Checks of declared mutations and state updates across successive invocations, as well as returned outputs.
- [ ] The Markdown document identifies the reference implementation, applicable accuracy criteria and test commands.

See [Reference accuracy criteria](#reference-accuracy-criteria) in the appendix. Data movement and integer or index outputs require exact comparison as appropriate to their semantics.

### Performance

For non-performance PRs, avoid unjustified performance regressions. New functionality may justify a reasonable end-to-end regression; quantify and explain it. Model performance may be reported as percentage changes rather than absolute figures.

- [ ] The PR description includes:
    - [ ] Operator performance results against the baseline.
    - [ ] Model performance results on a representative workload, identifying latency or throughput metrics.
    - [ ] Matching comparison conditions, including versions, devices, inputs, model configuration and measurement methodology.
    - [ ] Cold compilation, cache restoration and steady execution measured separately, with persistent workspace and graph-buffer memory costs accounted for.

### Clean Code

- [ ] Source files meet these requirements:
    - [ ] Clear responsibility boundaries, reasonable reuse, and comments explaining important design choices. See [coding guidelines](#tilelang-operator-coding-guidelines).
    - [ ] Clear naming consistent with surrounding code. Operator documents are named after the operator.
    - [ ] No silent changes to caller-visible dtype, layout, contents or external state. Permitted in-place behavior is documented and tested.
    - [ ] Compilation and execution errors retain their diagnostics.
- [ ] The PR description includes a recompilation risk assessment based on [JIT specialization and recompilation](#jit-specialization-and-recompilation):
    - [ ] **White-box analysis:** Parameter domains, compile-key fields, generated PrimFuncs and compiler options, with justification for compile-time choices.
    - [ ] **Black-box validation:** Observed shapes and compile keys on a representative model workload, with compilation counts separated by startup and serving phases.
    - [ ] Causes and mitigation of unnecessary variants, plus remaining compilation conditions.
- [ ] Where warmup is used:
    - [ ] Source files include the wrapper and registration site for required variants.
    - [ ] Test files cover key space, registration windows and runtime coverage.
    - [ ] The Markdown document describes warmup scope and related configuration.

Reduce unnecessary specialization before warming up required variants. Follow the upstream [vLLM JIT kernel warmup guide](https://docs.vllm.ai/en/latest/contributing/jit_kernel_warmup/) and the TileLang [operator integration](#operator-integration) requirements below.

## Appendix

### Reference accuracy criteria

- **Absolute tolerance (`atol`)**: a fixed error allowance, including when the reference is near zero.
- **Relative tolerance (`rtol`)**: an error allowance proportional to the absolute reference value.

Compare each output element `actual_i` with its reference `golden_i`:

$$
\left|\mathrm{actual}_i-\mathrm{golden}_i\right| \leq \mathrm{atol}+\mathrm{rtol}\cdot\left|\mathrm{golden}_i\right|
$$

| Output dtype | rtol | atol | Maximum absolute error |
| --- | --- | --- | --- |
| FP16 | $2^{-9}$ | $2^{-9}$ | 0.1 |
| BF16 | $2^{-6}$ | $2^{-6}$ | 1 |
| FP32 | $2^{-10}$ | $2^{-16}$ | 0.01 |

Both conditions must hold:

1. At least 99% of elements satisfy the combined tolerance.
2. The maximum absolute error across all elements does not exceed the corresponding limit.

The maximum-error rule is $\max\left(\mathrm{fixed\_limit},\;32\cdot\mathrm{ULP}_{1.0}\right)$, where $\mathrm{ULP}_{1.0}$ is the unit in the last place at 1.0 for the output dtype; use the explicit limits in the table.

### TileLang operator coding guidelines

#### Naming

- Use `snake_case` for functions, ordinary parameters and local variables.
- Use `UPPER_SNAKE_CASE` for constants. Choose descriptive names for buffers, dimensions and tuning parameters.
- Replace unexplained magic numbers with named constants or derived expressions. Explain the purpose and selection rationale of algorithm and tuning constants.

#### Public interface and integration

- Expose a Python wrapper; callers use it instead of compiling or launching a PrimFunc directly.
- Encapsulate input validation, output allocation, compile-key construction and tiling in the wrapper.
- Document supported shapes, dtypes, layouts and in-place behavior. Reject unsupported inputs or use an existing fallback.
- Use consistent compile-key and launch-configuration rules for warmup and runtime execution.
- Keep compiler imports inside worker-side methods. Use `HAS_TILELANG` from `vllm_ascend.ops.tilelang` for package detection and check hardware and operator support separately.

#### Interface structure example

The following pseudocode shows the interface boundaries; it is not a runnable operator implementation. The component retains one JIT wrapper per worker device.

```python
class YourOp:
    def __init__(self):
        self._kernel = YourOpJitKernel()

    def register_warmup(self, inputs):
        self._kernel.register_warmup(inputs)

    def __call__(self, input):
        output, key = ...
        self._kernel.get_kernel(key)(input, output)
        return output
```

Call `register_warmup(...)` during the active registry window, then use the same component's public entry point for runtime calls.

### JIT specialization and recompilation

`AscendTileLangJitKernel` caches executors by an explicit, immutable compile key. Include every input or option that changes the PrimFunc or compilation settings:

| Input or configuration | Compile-key requirement |
| --- | --- |
| Tensor shape, dtype or stride | Include values used to construct buffer types, indexing or generated code. |
| Tiling and algorithm choices | Include choices that change generated code. |
| Optional arguments | Include presence or mode when it changes the PrimFunc interface or implementation. |
| Compiler options | Include choices affecting pass configuration, compiler flags or the executor interface, such as `out_idx`. |

Do not add launch-time tensor contents or full addresses to the key unless the implementation embeds them into the compiled program. Source, compiler, target and cache-state changes can also affect TileLang's persistent artifact reuse; a wrapper cache hit and a persistent-cache load are distinct from compilation.

Validate key coverage against observed production inputs. After reducing variants, repeat accuracy and performance checks, then warm the required keys. Verify cold compilation and reuse from a second process with the same persistent cache when changing the compiler integration or dependencies.

### Operator integration

#### Implement the JIT wrapper

Subclass `AscendTileLangJitKernel` from `vllm_ascend.ops.tilelang.ascend_jit_warmup` and implement:

| Method | Purpose |
| --- | --- |
| `get_prim_func(key)` | Build the TileLang PrimFunc. Import TileLang inside this method. |
| `get_warmup_keys(...)` | Return the compile keys for the registered warmup inputs. |
| `get_compile_options(key)` (optional) | Supply `out_idx`, pass configuration or compiler flags. |

Use upstream TileLang's `tilelang.ascend.language` dialect. The wrapper selects `target="ascend"` and `execution_backend="tvm_ffi"`.

#### Register warmup and reuse the wrapper

Create one wrapper per worker device. Call `register_warmup(...)` inside the model runner's active `JitWarmupRegistry` window and register all reachable graph buckets. Registration outside an active registry window is a no-op. Use the same wrapper's `get_kernel(key)` for runtime execution.

The worker compiles registered TileLang kernels before ACL Graph capture. `compile(key)` builds an executor without running the operator. `get_kernel(key)` reuses the executor and rejects a compilation miss during ACL Graph capture.

#### Integrate execution

TileLang's TVM-FFI adapter installs the Torch NPU stream callback automatically and reads the current PyTorch NPU stream for each invocation. Validate non-default streams and changing inputs across graph replay.

Allocate persistent workspaces through `current_workspace_manager()` before memory profiling and graph capture, and retain their storage across replay. Test workspace reuse, output addresses and write boundaries, including first execution during capture when supported.

### A5 dependencies

Use the pinned TileLang and TVM FFI wheels from the A5 Dockerfiles and selected-test workflow; versions are listed in [Dependency Version Management](dependency_version_management.md). TileLang bundles TVM; do not install a separate `apache-tvm` package.

Keep the CANN environment, Bisheng compiler, CCE-capable linker, headers and runtime libraries available for worker-side JIT compilation. SHMEM communication support is tracked separately in the RFC.

### Testing

Run the wrapper and warmup unit tests:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/model_executor/warmup/test_jit_warmup_tilelang.py tests/ut/model_executor/warmup/test_kernel_warmup.py
```

In an A5 image with a visible NPU, run the infrastructure smoke test:

```bash
pytest -q tests/e2e/pull_request/one_card/test_tilelang_a5.py
```

These tests cover compilation, cache reuse, device ownership, stream and graph execution, workspace reuse, and FakeTensor integration. Each production operator also requires the operator and model coverage described above; follow the [PR testing requirements](testing.md#pr-testing-requirements) and [PR selective testing](testing.md#pr-selective-testing-ci) for validation and runner selection.
