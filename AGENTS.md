# vLLM Ascend Development Guidelines

This document provides instructions for contributors to the vLLM Ascend project. Please read and follow these guidelines to ensure code quality, maintainability, and consistency.

Read linked documentation when it applies to the current task, using the relevant sections. Continue authorized local implementation and validation while preparing the evidence needed for PR review.

---

## Table of Contents

- [Setup and Environment](#setup-and-environment)
    - [No New Environment Variables](#no-new-environment-variables)
    - [Configuration Review Requirement](#configuration-review-requirement)
- [Testing](#testing)
- [Code Style](#code-style)
    - [Python Conventions](#python-conventions)
    - [Naming Conventions](#naming-conventions)
- [Triton Operators](#triton-operators)
- [NPU-Specific Considerations](#npu-specific-considerations)
    - [Tensor item() Operations](#tensor-item-operations)
    - [Memory and Performance](#memory-and-performance)
- [Model and Plugin Architecture](#model-and-plugin-architecture)
    - [vLLM Ascend Plugin Architecture](#vllm-ascend-plugin-architecture)
    - [Patching Requirement](#patching-requirement)
    - [Model Runner Changes](#model-runner-changes)
- [Release Notes](#release-notes)
- [Commit Messages and Pull Requests](#commit-messages-and-pull-requests)
    - [Commit Message Format](#commit-message-format)
- [Quick Start for Contributors](#quick-start-for-contributors)
- [References](#references)

---

## Setup and Environment

### No New Environment Variables

Do not add new environment variables. User-facing configuration belongs in vLLM's additional config mechanism: a typed key under `--additional-config` (offline: `LLM(additional_config={...})`), following the existing sub-config patterns in [vllm_ascend/ascend_config.py](vllm_ascend/ascend_config.py) and documented in [Additional Configuration](docs/source/user_guide/configuration/additional_config.md). The project has already migrated most configuration from environment variables to additional config (see the [migration guide](docs/source/user_guide/configuration/additional_config.md#migration-guide)); do not reverse it.

An environment variable is a last resort, only when additional config cannot cover the case — for example, a value needed before vLLM config exists (process bootstrap, subprocess launch). A strong reason is required; "it is simpler" is not one.

When an environment variable is truly unavoidable, define it in the centralized `env_variables` dictionary in [vllm_ascend/envs.py](vllm_ascend/envs.py).

**Requirements:**

- Add documentation for each environment variable in the `env_variables` dict comment
- Specify default values and valid ranges
- Indicate whether the variable is sensitive (credentials, keys)

**Never**: Hardcode environment variable names throughout the codebase. Reference them from the central module using `from vllm_ascend import envs`.

### Configuration Review Requirement

Both configuration channels are strictly managed. The PR must justify the channel it chooses.

For a **new additional config key** (the recommended channel), the PR must show:

- Why existing configuration cannot cover the use case
- Typed parsing and validation in `vllm_ascend/ascend_config.py`, following the existing sub-config pattern (e.g. `EplbConfig`, `RlConfig`), not raw dict access scattered across the code
- A table entry in [Additional Configuration](docs/source/user_guide/configuration/additional_config.md) with type, default, and behavior
- A name that generalizes beyond the model or workload that motivated it

For a **new environment variable** (exceptional), the PR must show a strong reason on top:

- Why additional config cannot cover the use case
- The variable name follows the `VLLM_ASCEND_*` naming convention
- The default value works for all supported hardware
- Documentation is added to the `env_variables` dict and reflected in the user-facing [environment variables](docs/source/user_guide/configuration/env_vars.md) page

---

## Testing

Run tests appropriate to the change and complete required checks. Broaden or repeat testing when new changes, failures, or unresolved risks justify it.

When preparing or reviewing a PR, read the [PR testing requirements](docs/source/developer_guide/contribution/testing.md#pr-testing-requirements) for coverage, hardware evidence, and validation reporting. Use the other sections of the same guide for environment setup and test commands as needed.

**What you can verify without NPU hardware:**

- Unit tests run on CPU: `TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut` (commands and mocking conventions in the [Unit tests](docs/source/developer_guide/contribution/testing.md#unit-tests) section)
- Format and lint checks: `bash format.sh ci`
- Documentation build and link checks, when docs changed

E2E, multi-card, and performance tests require real NPU hardware. Do not attempt them in a CPU-only environment; delegate that coverage to CI and say so in the PR.

---

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Type-checking imports wrapped in `if TYPE_CHECKING:`

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE`
    - Immutable configuration objects

    If mutable module-level state is necessary, explain its lifetime and concurrency implications in the PR.

- **No Magic Numbers**: Use named constants with descriptive names.

- **Descriptive Naming**: Use names that describe functionality, not implementation details.

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NPUModelRunner`, `ACLGraphManager`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`, `compute_attention`)
- **Constants**: `ALL_UPPER_CASE`
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

---

## Triton Operators

When adding, modifying, or reviewing Triton operators, follow the [Triton operator contribution guidelines](docs/source/developer_guide/contribution/triton_operators.md) for deliverables, functionality, accuracy, performance, and code quality.

Contributors and reviewers must check:

- The documented interface, production integration, and compatibility across applicable devices.
- Operator and model accuracy and performance evidence, with test coverage appropriate to framework-wide or model-specific use.
- Public wrapper usage, documented side effects, and JIT specialization and recompilation risks.
- When warmup is used, required variants, registration in the selected component's active registry window, and reuse of the same wrapper for runtime calls.

For warmup integration or changes, read [Triton JIT warmup on Ascend](docs/source/developer_guide/Design_Documents/triton_jit_warmup.md) for the startup flow, registration locations, and validation steps.

---

## NPU-Specific Considerations

### Tensor item() Operations

**Warning**: `tensor.item()` operations cause synchronization overhead on NPU when the `tensor` is on device.

If the `tensor` is a device tensor, calling `item()` triggers a synchronous data transfer from NPU to CPU. This can severely degrade performance in hot paths, causing `AsyncScheduler` to block here.

**Review Requirements:**

1. Profile synchronization introduced in hot paths before merging
2. Consider alternative patterns:
    - Keep values on device when possible
    - Batch operations to reduce sync frequency
    - Use device-side operations (e.g., `torch.argmax`, `torch.sum`)
3. Document when `item()` is unavoidable (e.g., logging, conditional logic)

### Memory and Performance

Additional NPU-specific best practices:

- Avoid CPU-NPU memory transfers in hot paths
- Prefer in-place operations where safe (e.g., `x.add_()`, `x.mul_()`)
- Monitor memory fragmentation, especially for long-running processes
- Test with realistic workloads on actual NPU hardware (e.g. Ascend 910B, 910C, 310P)

---

## Model and Plugin Architecture

### vLLM Ascend Plugin Architecture

vLLM Ascend is a **hardware plugin** that integrates with upstream vLLM via the pluggable hardware interface. Prefer existing upstream implementations and extension points. Ascend-specific model adaptations are also registered in the [model registry](vllm_ascend/models/__init__.py); follow existing registration and inheritance patterns when working on these models.

Choose the integration approach that fits the change:

1. **Patching** (in `vllm_ascend/patch/`):
    - `vllm_ascend/patch/platform/` - Platform-level patches (distributed, scheduling)
    - `vllm_ascend/patch/worker/` - Worker-level patches (model-specific behavior)
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - `NPUModelRunner(GPUModelRunner)` - Extend vLLM model runner with NPU-specific behavior
    - Add NPU-specific components via composition (e.g., `AclGraphManager`)
    - Custom Operators - NPU-specific custom operators (e.g., `AscendRMSNorm`)

3. **External upstream contributions** where appropriate

When adding or adapting a model, follow the step-by-step walkthrough in the [vllm-ascend-model-adapter skill](.agents/skills/vllm-ascend-model-adapter/SKILL.md).

### Patching Requirement

Explain new patches in the PR for architectural review.

When adding or changing a patch, read the [patch strategy](docs/source/developer_guide/Design_Documents/patch.md) for lifecycle, registration, and removal-plan requirements.

Your PR must explain:

- Which upstream component the patch targets and why
- How the patch stays minimal and focused
- The performance implications
- The long-term plan for upstream contribution

### Model Runner Changes

Explain new `model_runner` behavior in the PR for architectural review.

Your PR must explain:

- Why existing extension points cannot cover the new behavior
- The performance implications on NPU hardware
- Compatibility with existing model implementations
- Long-term maintainability and test coverage

**NPU Model Runner Files:**

- `vllm_ascend/worker/v2/model_runner.py` - vLLM v2 model runner

---

## Release Notes

Every PR with a user-facing change must ship a release note fragment in the same PR. The
fragment is written for users of the released wheel or Docker image, not for reviewers:
a reader who never sees the diff should be able to tell what changed for them and whether
they need to do anything.

**Write a fragment for**: new models, hardware, or operators; new features; changes to
user-facing APIs, parameters, environment variables, or configuration; breaking changes,
deprecations, and default behavior changes; dependency and toolchain updates (CANN,
torch_npu, triton-ascend, vLLM alignment); performance improvements; and bug fixes users
could have hit (hangs, OOM, wrong outputs, broken metrics).

**Do not write a fragment for**: internal refactors, code moves, logging and typing
cleanups; tests, CI, build scripts, development tooling; documentation typo fixes and
small edits (guides for brand-new scenarios are the exception, category `documentation`);
and plumbing for future features that is still invisible to users. If the only honest
sentence is "refactored X" or "prepared Y for Z", users cannot observe the change — no
fragment.

**How to write the text** (full rules and examples in
[release_notes/README.md](release_notes/README.md#writing-the-text)):

- Say what the user gets, not what the code does. Name user-facing switches exactly (env
  vars, config keys, CLI flags); skip file, class, and kernel-level narration.
- Performance entries: state the scenario (model, workload, hardware) and the effect
  (measured metric and magnitude, or the concrete user-visible mechanism). Never invent
  numbers.
- Breaking and config entries: old behavior → new behavior, the exact switch, the
  migration step, and what happens if the user does nothing.
- Bug fixes: the user-visible symptom and when it occurs, not the internal root cause.
- The unit of a release note is the net change relative to the previous release. A fix for
  a bug **already shipped in a released version** needs a `bugfix` fragment; a fix for a
  bug that never reached a released artifact needs no fragment of its own. A revert of a PR
  merged after the last release needs no fragment either. In both unreleased cases, remove
  the cancelled PR's not-yet-archived fragment in the fix/revert PR. See
  [release_notes/README.md](release_notes/README.md#changes-that-cancel-out-before-a-release).

Add `release_notes/fragments/<PR number>.yaml` following `release_notes/TEMPLATE.yaml` — open the PR first to get its number, then push the fragment to the same PR. When in
doubt about whether users can observe the change, add a fragment — the release manager drops
extra entries, but a missing one is lost. Reviewers confirm the inclusion decision and the
wording together with the code; there is no CI gate.

For release management workflows (release checklist issues, critical bug triage, fragment aggregation), use the [vllm-ascend-release skill](.agents/skills/vllm-ascend-release/SKILL.md).

---

## Commit Messages and Pull Requests

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format and **must include a sign-off**:

```bash
git commit -s -m "<type>: <summary>" -m "<body - explaining what changed and why>"
```

**Valid Types**: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

Follow the [commit message content guidelines](docs/source/developer_guide/contribution/index.md#commit-message-content) for the problem, fix, and validation.

### Pull Request Title Format

PR titles use bracketed change-type tags such as `[BugFix]`, `[Feature]`, or `[Doc]` (note: this differs from the `type:` prefix used in commit messages). The full accepted set is enforced by the CI check linked in the [PR submission checklist](docs/source/developer_guide/contribution/testing.md#pr-submission-checklist).

### Pull Request Template

When creating a PR, please follow the template in `.github/PULL_REQUEST_TEMPLATE.md` and ensure the following sections are completed:

> **Note**: The PR description will be automatically updated by GitHub Actions to include vLLM version info at the bottom. If you update the PR description via API or CLI, make sure to preserve the `- vLLM version:` and `- vLLM main:` lines.

- **What this PR does / why we need it?** - Clearly describe the changes and their purpose
- **Does this PR introduce _any_ user-facing change?** - Indicate if there are any user-visible changes
- **How was this patch tested?** - Describe how you tested the changes. Examples:
    - Unit tests added/updated: list the test files
    - Manual testing: provide the test steps and commands
    - CI testing: indicate if only CI verification is needed

    **Only report test runs you actually performed.** Without NPU hardware, write "CI testing only" and let CI produce the hardware evidence — never claim NPU or performance results that were not produced.

If the change renames or deletes files, update `.github/CODEOWNERS` in the same PR; entries pointing at removed paths fail the CI gate for everyone (see the [CODEOWNERS approval gate](docs/source/developer_guide/contribution/testing.md#codeowners-approval-gate)).

Document public APIs and reflect user-facing behavior changes in the project docs.

---

## Quick Start for Contributors

For environment setup, use the [contribution guide](docs/source/developer_guide/contribution/index.md#building-and-testing) as needed.

Before pushing, run the required checks for all file types, including Markdown:

```bash
pip install -r requirements-lint.txt  # first-time setup
bash format.sh ci
```

Inspect any formatter changes and include them in the commit.

If `format.sh ci` fails, fix real lint and format errors. Exception: the `gitleaks-offline-scan` hook can fail locally for environment or git-history reasons even when no secret was added — if gitleaks is the only failure and the diff adds no secrets, proceed and let CI confirm.

Push to your fork and create the PR against the upstream repository, following the commit and PR requirements above.

---

## References

- [vLLM Hardware Plugin RFC](https://github.com/vllm-project/vllm/issues/11162)
- [Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Contributors Guide](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)
