# vLLM Ascend Development Guidelines

This document provides instructions for contributors to the vLLM Ascend project. Please read and follow these guidelines to ensure code quality, maintainability, and consistency.

---

## Table of Contents

- [Setup and Environment](#setup-and-environment)
    - [Environment Variables](#environment-variables)
    - [Environment Variable Review Requirement](#environment-variable-review-requirement)
- [Testing](#testing)
    - [Unit and System Tests](#unit-and-system-tests)
    - [Running Tests](#running-tests)
- [Code Style](#code-style)
    - [Python Conventions](#python-conventions)
    - [Naming Conventions](#naming-conventions)
- [NPU-Specific Considerations](#npu-specific-considerations)
    - [Tensor item() Operations](#tensor-item-operations)
    - [Memory and Performance](#memory-and-performance)
- [Model and Plugin Architecture](#model-and-plugin-architecture)
    - [vLLM Ascend Plugin Architecture](#vllm-ascend-plugin-architecture)
    - [Patching Requirement](#patching-requirement)
    - [Model Runner Changes](#model-runner-changes)
- [Commit Messages and Pull Requests](#commit-messages-and-pull-requests)
    - [Commit Message Format](#commit-message-format)
- [Review Checklist](#review-checklist)
    - [Code Quality](#code-quality)
    - [Testing](#testing-1)
    - [Documentation](#documentation)
    - [NPU Considerations](#npu-considerations)
    - [Commit and PR](#commit-and-pr)
- [Quick Start for Contributors](#quick-start-for-contributors)
- [References](#references)

---

## Setup and Environment

### Environment Variables

All environment variables must be defined in `vllm_ascend/envs.py` using the centralized `env_variables` dictionary.

**Requirements:**

- Add documentation for each environment variable in the `env_variables` dict comment
- Specify default values and valid ranges
- Indicate whether the variable is sensitive (credentials, keys)

**Example:**

```python
env_variables = {
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # ...
}
```

**Never**: Hardcode environment variable names throughout the codebase. Reference them from the central module using `from vllm_ascend import envs`.

### Environment Variable Review Requirement

**Strict Review Required**: All new environment variables must undergo code review.

Reviewers must verify:

- The variable name follows the `VLLM_ASCEND_*` naming convention
- Default value is appropriate for all supported hardware
- Documentation is added to the `env_variables` dict
- The variable is used in a performance-critical path

---

## Testing

### Unit and System Tests

**Requirement**: All new functionality requires corresponding tests.

- **Unit Tests (UT)**: Located in `tests/ut/`, cover core logic, edge cases, and error conditions
- **System Tests (ST)**: Located in `tests/e2e/`, verify end-to-end behavior and integration points
- **Nightly Tests**: Include benchmarks for NPU-specific code paths in `tests/e2e/nightly/`

**Test Coverage Guidelines:**

- New features: Tests must cover happy path and failure modes
- Bug fixes: Tests must include a regression test for the bug
- Performance-critical code: Include benchmarks and performance regression tests

### Running Tests

```bash


# Run specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run NPU-specific tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/test_piecewise_res_consistency
```

**Requirement**: Run all tests locally before requesting review. Verify tests pass on NPU hardware for NPU-specific changes.

---

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Type-checking imports wrapped in `if TYPE_CHECKING:`

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE` in `envs.py`)
    - Immutable configuration objects

    **Requires Approval:**
    - Any new mutable global state

- **No Magic Numbers**: Use named constants with descriptive names:

    ```python
    # Bad
    if seq_len > 2048: ...

    # Good
    MAX_CONTEXT_LENGTH = 2048
    if seq_len > MAX_CONTEXT_LENGTH: ...
    ```

- **Descriptive Naming**: Use names that describe functionality, not implementation details.

    ```python
    # Bad
    is_deepseek_v3_r1
    flag1
    tmp_var

    # Good
    supports_dynamic_temperature
    uses_speculative_decoding
    ```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NPUModelRunner`, `AscendSampler`, `ACLGraphManager`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`, `compute_attention`)
- **Constants**: `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE`, `VLLM_ASCEND_ENABLE_NZ`)
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

---

## NPU-Specific Considerations

### Tensor item() Operations

**Warning**: `tensor.item()` operations cause synchronization overhead on NPU when the `tensor` is on device.

If the `tensor` is a device tensor, the operator `item()` will trigger a synchronous data transfer from NPU to CPU, which can severely degrade performance in hot paths, cause this will make `AsyncScheduler` block here.

**Review Requirements:**

1. Profile performance impact before merging
2. Consider alternative patterns:
    - Keep values on device when possible
    - Batch operations to reduce sync frequency
    - Use device-side operations (e.g., `torch.argmax`, `torch.sum`)
3. Document when `item()` is unavoidable (e.g., logging, conditional logic)

**Example Patterns:**

```python
# Bad: In hot loop - causes sync per iteration
for tensor in tensors:
    value = tensor.item()

# Better: Batch operations - single sync
values = [t.item() for t in tensors]  # Single batch sync

# Good: Keep on device when possible
max_value = torch.max(tensor)  # No sync needed
if max_value > threshold:  # Comparison can stay on device
    ...
```

### Memory and Performance

Additional NPU-specific best practices:

- Avoid CPU-NPU memory transfers in hot paths
- Prefer in-place operations where safe (e.g., `x.add_()`, `x.mul_()`)
- Monitor memory fragmentation, especially for long-running processes
- Test with realistic workloads on actual NPU hardware (Ascend 910B/C)

---

## Model and Plugin Architecture

### vLLM Ascend Plugin Architecture

vLLM Ascend is a **hardware plugin** that integrates with upstream vLLM via the pluggable hardware interface. It does not add new model files directly.

**Required Pattern**: Model-specific functionality should be implemented via:

1. **Patching** (in `vllm_ascend/patch/`):
    - `vllm_ascend/patch/platform/` - Platform-level patches (distributed, scheduling)
    - `vllm_ascend/patch/worker/` - Worker-level patches (model-specific behavior)
    - Example: `patch_deepseek.py` modifies upstream Deepseek model behavior
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - `NPUModelRunner(GPUModelRunner)` - Extend vLLM model runner with NPU-specific behavior
    - `AscendSampler` - Extend vLLM sampler with NPU-specific operations
    - Add NPU-specific components via composition (e.g., `AclGraphManager`)
    - Custom Operators - NPU-specific custom operators (e.g., `AscendRMSNorm`)

3. **External upstream contributions** where appropriate

### Patching Requirement

**Strict Review Required**: All new patches must undergo thorough architectural review.

Reviewers must verify:

- The patch targets the correct upstream component
- The patch is minimal and focused
- Performance implications are understood
- A long-term plan exists for upstream contribution

**Example Patch Pattern:**

```python
# vllm_ascend/patch/worker/patch_deepseek.py
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

def forward(self, input_ids, positions, ...):
    # NPU-specific forward implementation
    ...

DeepseekV2Model.forward = forward  # Patch upstream class
```

### Model Runner Changes

**Strict Review Required**: All new behaviors added to `model_runner` must undergo thorough architectural review.

Reviewers must verify:

- The necessity of the new behavior (why can't this be in a patch?)
- Performance implications on NPU hardware
- Compatibility with existing model implementations
- Long-term maintainability and test coverage

**NPU Model Runner Files:**

- `vllm_ascend/worker/model_runner_v1.py` - vLLM v1 model runner
- `vllm_ascend/worker/v2/model_runner.py` - vLLM v2 model runner
- `vllm_ascend/_310p/model_runner_310p.py` - Ascend 310P model runner

---

## Commit Messages and Pull Requests

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```txt
<type>: <summary>

<body - explaining what changed and why>
```

**Valid Types**: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

**Good Examples:**

```txt
feat(npu): add flash attention support for Ascend CANN

- Implements FlashAttention-2 kernel for NPU backend
- Reduces memory usage by 30% compared to baseline

fix(model_runner): correct padding token handling

- Fixes token padding that caused incorrect attention masks
- Addresses issue #1234

perf: avoid CPU-NPU sync in attention computation

- Inline computation to avoid tensor.item() calls
- Improves throughput by 15%
```

**Bad Examples:**

```txt
fix bug
add feature
update code
```

---

## Review Checklist

Before merging, verify:

### Code Quality

- [ ] Code follows style guidelines (naming, imports, no magic numbers)
- [ ] No global state added without justification
- [ ] Patching pattern used correctly (if applicable)
- [ ] No direct model file additions

### Testing

- [ ] New tests added for new functionality (`tests/ut/` or `tests/e2e/`)
- [ ] Existing tests pass
- [ ] NPU-specific tests verified on actual hardware
- [ ] Performance benchmarks included where applicable

### Documentation

- [ ] Environment variables documented
- [ ] Public APIs documented
- [ ] User-facing changes reflected in docs

### NPU Considerations

- [ ] `tensor.item()` usage reviewed for performance impact
- [ ] No unnecessary CPU-NPU transfers in hot paths
- [ ] Memory usage verified on NPU hardware

### Commit and PR

- [ ] Commit messages are clear and descriptive
- [ ] PR description is complete
- [ ] All review comments addressed

---

## Quick Start for Contributors

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest tests/`
3. Check linting: `ruff check vllm_ascend/`
4. Format code: `ruff format vllm_ascend/`
5. Make your changes following guidelines in this document
6. Add tests for new behavior
7. Run full test suite before committing
8. Create a PR with clear description

---

## References

- [vLLM Hardware Plugin RFC](https://github.com/vllm-project/vllm/issues/11162)
- [Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Contributors Guide](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)
