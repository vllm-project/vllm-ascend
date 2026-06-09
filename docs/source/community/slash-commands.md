# Slash Commands

vLLM Ascend supports slash commands in pull request comments to trigger CI workflows. See the [Permission](#permission) section for who can trigger each command.

## Available Commands

### `/e2e`

Run specific E2E tests under `tests/e2e/pull_request/`. Tests are automatically routed to the appropriate NPU runner based on the test path.

**Examples:**

```text
# Run a single test on the default runner (a2 single card)
/e2e tests/e2e/pull_request/one_card/test_attention.py

# Run multiple tests across different runners
/e2e tests/e2e/pull_request/one_card/test_attention.py tests/e2e/pull_request/two_card/test_parallel.py

# Run tests on 310P
/e2e tests/e2e/pull_request/one_card/_310p/test_310p_ops.py
```

**Routing rules** (matched in order):

| Test path contains | Runner |
|---|---|
| `four_card/_310p` | 310P 4-card |
| `_310p` (under `one_card`/`two_card`) | 310P single card |
| `four_card` | A3 4-card |
| `two_card` | A3 2-card |
| Others (e.g. `one_card`) | A2 single card |

> Only test paths under `tests/e2e/pull_request/` are supported. Tests in `tests/e2e/schedule/` or `tests/e2e/doctests/` are not accepted by `/e2e`. Periodic (nightly / weekly) tests run via the [Periodic-Test workflow](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/schedule_periodic_test.yaml) — see [Nightly CI Test](../developer_guide/contribution/nightly_ci_test.md).

Tests are run against both the community vLLM version and the latest release.

### `/rerun`

Re-run all failed workflow runs on the current PR commit. Useful when CI jobs failed due to infrastructure issues.

**Examples:**

```text
# Re-run all failed CI workflows on this PR
/rerun
```

### `/nightly`

Trigger registered periodic nightly cases on the current PR commit. Cases are selected
from `nightly-main` in `.github/workflows/scripts/schedule_config.yaml` and matched with
the same `test_filter` rules used by the `Periodic-Test` workflow.

**Examples:**

```text
# Run all nightly-main cases
/nightly all

# Run one or more filtered nightly cases
/nightly Qwen3-8B test_fused_moe

# Use images from a non-main branch tag while testing the PR commit
/nightly Qwen3-8B --branch release/v0.11
```

Only users with triage+ permission can trigger `/nightly`.

## Behavior

1. When you comment a slash command, a 👀 reaction is added to your comment to indicate it has been received
2. The corresponding CI workflow is triggered asynchronously
3. Upon completion, a 🎉 reaction and a summary comment are added

## Scope

| Command | PR comments | Issue comments |
|---|---|---|
| `/e2e` | ✅ | ❌ |
| `/nightly` | ✅ | ❌ |
| `/rerun` | ✅ | ❌ |

## Permission

| Command | Who can trigger |
|---|---|
| `/e2e` | PR author, or users with triage+ permission on the repository |
| `/nightly` | Users with triage+ permission on the repository |
| `/rerun` | PR author, or users with triage+ permission on the repository |

Permission is verified via the GitHub API (`repos/{owner}/{repo}/collaborators/{user}/permission`).
