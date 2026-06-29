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

> Only test paths under `tests/e2e/pull_request/` are supported. Tests in `tests/e2e/nightly/`, `tests/e2e/models/`, or `tests/e2e/doctests/` are not accepted by `/e2e`. Use `/nightly` for nightly tests.

Tests are run against both the community vLLM version and the latest release.

### `/nightly`

Trigger specific nightly test cases on A2 and A3. Supports only PR comments. Test case names correspond to the `test_config.name` entries defined in `schedule_nightly_test_a2.yaml` and `schedule_nightly_test_a3.yaml`.

**Usage:**

| Syntax | Scope |
|---|---|
| `/nightly <test_cases>` | Runs on `main` branch |
| `/nightly <test_cases> --branch <branch>` | Runs on the specified branch |

Use `--branch <name>` to specify a target branch. Without `--branch`, all arguments are treated as test cases (separated by commas or spaces) and the branch defaults to `main`.

> **Note**: When commenting on a PR, the tests run on the PR branch automatically in the triggered workflow; the `--branch` flag is primarily used in issue comments.

**Common test case names (A2):**

`test_custom_op`, `test_custom_op_multi_card`, `qwen3-vl-32b-instruct-w8a8`, `qwen3-32b-int8`, `MiniMax-M2.5-w8a8-QuaRot-A2`, `Qwen3.5-27B-w8a8-A2`, `Qwen3.5-397B-A17B-w4a8-mtp`, `accuracy-group`

**Common test case names (A3):**

`multi-node-deepseek-v3.2-W8A8-EP`, `mtpx-deepseek-r1-0528-w8a8`, `deepseek-r1-0528-w8a8`, `kimi-k2-thinking`, `qwen3-vl-235b-a22b-instruct-w8a8`, `custom-multi-ops`, ...

**Examples:**

```text
# Run a single test case on main branch
/nightly qwen3-vl-32b-instruct-w8a8

# Run on a specific release branch
/nightly qwen3-vl-32b-instruct-w8a8 --branch releases/v0.23.0

# Run all tests on a specific branch
/nightly all --branch my-feature-branch

# Run multiple test cases (comma-separated)
/nightly test_custom_op,multi-node-deepseek-v3.2-W8A8-EP

# Run multiple test cases (space-separated, also works)
/nightly test_custom_op accuracy-group

# Run accuracy group tests (branch defaults to main)
/nightly accuracy-group
```

This triggers `workflow_dispatch` on both `schedule_nightly_test_a2.yaml` and `schedule_nightly_test_a3.yaml`.

### `/cherry-pick`

Cherry-pick a PR's commits onto a specified target branch and create a new PR. This is useful for backporting fixes to release branches.

**Usage:**

| Syntax | Description |
|---|---|
| `/cherry-pick <target_branch>` | Cherry-pick onto the specified branch |

**Examples:**

```text
# Cherry-pick to a release branch
/cherry-pick releases/v0.23.0

# Cherry-pick to main
/cherry-pick main
```

A new PR will be created with the title format `[Cherry-pick] <original_title> (from #<PR_NUMBER>)` and a body linking back to the original PR.

If the cherry-pick encounters merge conflicts, the command will report the failure and the cherry-pick must be done manually.

### `/rerun`

Re-run all failed workflow runs on the current PR commit. Useful when CI jobs failed due to infrastructure issues.

**Examples:**

```text
# Re-run all failed CI workflows on this PR
/rerun
```

## Behavior

1. When you comment a slash command, a 👀 reaction is added to your comment to indicate it has been received
2. The corresponding CI workflow is triggered asynchronously
3. Upon completion, a 🎉 reaction and a summary comment are added

## Scope

| Command | PR comments | Issue comments |
|---|---|---|
| `/e2e` | ✅ | ❌ |
| `/rerun` | ✅ | ❌ |
| `/cherry-pick` | ✅ | ❌ |
| `/nightly` | ✅ | ❌ |

## Permission

| Command | Who can trigger |
|---|---|
| `/e2e` | PR author, or users with triage+ permission on the repository |
| `/rerun` | PR author, or users with triage+ permission on the repository |
| `/cherry-pick` | PR author, or users with triage+ permission on the repository |
| `/nightly` | Users with triage+ permission on the repository only |

Permission is verified via the GitHub API (`repos/{owner}/{repo}/collaborators/{user}/permission`).
