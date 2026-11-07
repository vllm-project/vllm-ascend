# Contributing Triton operators

## Deliverables checklist

- [ ] PR description.
- [ ] Source files.
- [ ] Markdown operator document: `vllm_ascend/ops/triton/docs/<operator>.md`.
- [ ] Test files: `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`; warmup unit tests, when applicable: `tests/ut/model_executor/warmup/`.
- [ ] Repository-wide deliverables: see the [contribution guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html) and [release note guide](https://github.com/vllm-project/vllm-ascend/blob/main/release_notes/README.md).

## Overall objectives

Triton operators merged into the project should meet quality requirements in four areas:

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
    - [ ] Cross-device compatibility and actual validation results. Operators should not be restricted to one particular device; device-specific optimization preserves functionality and correctness on other applicable devices.
- [ ] The Markdown document follows the [operator template](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/docs/triton_description_specification.md), describing functionality, algorithm, parameters, constraints, origin and differences, and test entry points.
- [ ] The implementation matches the documented inputs, outputs, and behavior.
- [ ] The operator is integrated into the corresponding model or shared operator call path in vLLM-Ascend.

### Accuracy

- [ ] The PR description includes:
    - [ ] Operator accuracy results against a reference implementation.
    - [ ] Model accuracy results against the baseline on an evaluation dataset such as GPQA, with evaluation settings.
- [ ] Test files provide:
    - [ ] Coverage of all declared input categories and relevant boundaries.
    - [ ] For framework-wide operators, coverage of the general interface and representative real shapes from applicable models; one model's shapes alone do not establish general coverage.
    - [ ] For model-specific operators, coverage of real shapes and parameter combinations reachable in the target models under the declared configurations.
    - [ ] Parameterized representative cases and boundaries, avoiding redundant cases.
- [ ] The Markdown document identifies the reference implementation, applicable accuracy criteria, and test commands.

See the appendix for reference accuracy criteria. Data movement and integer or index outputs require exact comparison as appropriate to their semantics.

### Performance

For non-performance PRs, avoid unjustified performance regressions. New functionality may justify a reasonable end-to-end regression; quantify and explain it. Model performance may be reported as percentage changes rather than absolute figures.

- [ ] The PR description includes:
    - [ ] Operator performance results against the baseline.
    - [ ] Model performance results on a workload such as GSM8K, identifying latency or throughput metrics.
    - [ ] Matching comparison conditions, including versions, devices, inputs, model configuration, and measurement methodology.

### Clean Code

- [ ] Source files meet these requirements:
    - [ ] Clear responsibility boundaries, reasonable reuse, and comments explaining important design choices. See the operator coding guidelines in the appendix for naming, interface boundaries, and kernel behavior.
    - [ ] Clear naming for public entry points, private kernels, warmup wrappers, parameters, constants, and tests, consistent with surrounding code. Operator documents are named after the operator and follow the [repository layout](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/README.md).
    - [ ] No silent changes to caller-visible dtype, layout, contents, or external state. Permitted in-place behavior is documented and tested.
- [ ] The PR description includes a recompilation risk assessment:
    - [ ] **White-box analysis:** Parameter domains, specialization triggers, and justification for annotations such as constexpr and specialization controls.
    - [ ] **Black-box validation:** Observed shapes and specialization metadata on a workload such as GPQA, with compilation counts separated by startup and serving phases.
    - [ ] Causes and mitigation of unnecessary variants, plus remaining compilation conditions.
- [ ] Where warmup is used:
    - [ ] Source files include the wrapper and registration site for required variants.
    - [ ] Test files cover key space, registration windows, and runtime coverage.
    - [ ] The Markdown document describes warmup scope and related configuration.

Reduce unnecessary specialization first. Where appropriate, use warmup to move compilation of required variants before serving; it does not replace variant control. See the upstream [vLLM JIT kernel warmup guide](https://docs.vllm.ai/en/latest/contributing/jit_kernel_warmup/) for the shared framework and integration approach.

## Appendix

### Reference accuracy criteria

For each element, apply the combined tolerance:

`abs(actual_i - golden_i) <= atol + rtol * abs(golden_i)`

| Output dtype | rtol | atol | Maximum absolute error |
| --- | --- | --- | --- |
| FP16 | 2^-9 | 2^-9 | 0.1 |
| BF16 | 2^-6 | 2^-6 | 1 |
| FP32 | 2^-10 | 2^-16 | 0.01 |

Both conditions must hold:

1. At least 99% of elements satisfy the combined tolerance.
2. The maximum absolute error across all elements does not exceed the corresponding limit.

`atol` controls error near zero, while `rtol` controls error relative to the reference magnitude. The maximum-error rule is `max(fixed_limit, 32 * ULP_1.0)`, where `ULP_1.0` is the unit in the last place at 1.0 for the output dtype; use the explicit limits in the table.

### Triton operator coding guidelines

#### Naming

- Use `snake_case` for functions, ordinary parameters, and local variables, for example `hidden_size` and `num_tokens`.
- Use `UPPER_SNAKE_CASE` for constants and compile-time parameters, for example `BLOCK_SIZE`. Use `tl.constexpr` only where compile-time specialization is needed; uppercase naming alone does not make a parameter a compile-time constant.
- For Tensor arguments of a Triton kernel, prefer `xxx_ptr` names, for example `input_ptr` and `output_ptr`, reflecting their use as pointers inside the kernel.
- Replace unexplained magic numbers with named constants or derived expressions. Explain the purpose and selection rationale of algorithm and tuning constants.

#### Public interface and integration

- Expose a Python wrapper as the operator's public interface. Model and shared operator code call this wrapper rather than launching the internal kernel directly.
- Keep input handling, output allocation or validation, implementation selection, grid calculation, and tiling configuration inside the wrapper or its internal helpers. Callers provide semantic inputs rather than kernel launch details.
- Document inputs, outputs, supported shapes, dtypes, layout constraints, and any in-place behavior. Check necessary preconditions at an appropriate boundary, avoiding repeated expensive checks on the execution path.
- Handle unsupported inputs explicitly through an error or an existing fallback. Do not silently return incorrect results.
- Keep warmup and runtime launch configuration consistent so that warmup covers the variants used by the public interface.

#### Interface structure example

The following pseudocode shows the interface boundaries; it is not a runnable operator implementation.

```python
@triton.jit
def _your_op_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    ...


def your_op(input):
    # Validate inputs and prepare the output.
    ...
    # Select tiling and calculate the grid internally.
    ...
    _your_op_kernel[grid](
        input, output, num_elements, BLOCK_SIZE=block_size
    )
    return output
```

Callers import and use the public entry point:

```python
from path.to.your_op import your_op

output = your_op(input)
```
