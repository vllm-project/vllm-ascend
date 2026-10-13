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
| `two_card/_310p` | 310P 2-card |
| `one_card/_310p` | 310P single card |
| `four_card` | A3 4-card |
| `two_card` | A3 2-card |
| Others (e.g. `one_card`) | A2 single card |

> Only test paths under `tests/e2e/pull_request/` are supported. Tests in `tests/e2e/nightly/`, `tests/e2e/models/`, or `tests/e2e/doctests/` are not accepted by `/e2e`. Use `/nightly` for nightly tests.

For doctests, run the **Doc Test** workflow manually in GitHub Actions, or let a relevant PR change trigger it automatically. See [Run doctest](../developer_guide/contribution/testing.md#run-doctest) for supported cases and local commands.

Tests are run against both the community vLLM version and the latest release.

### `/nightly`

Trigger specific nightly test cases on A2 and A3. Supports only PR comments. Test case names correspond to the `test_config.name` entries defined in `schedule_nightly_test_a2.yaml` and `schedule_nightly_test_a3.yaml`.

**Usage:**

| Syntax | Scope |
|---|---|
| `/nightly <test_cases>` | Runs on `main` branch |
| `/nightly <test_cases> --branch <branch>` | Runs on the specified branch |
| `/nightly <test_cases> --aop_enabled` | Enable AOP hooks (bisect / classify) on failure |
| `/nightly <test_cases> --vllm-ref <ref>` | Install the given vllm commit / tag / branch instead of the image-baked vllm |

Use `--branch <name>` to specify a target branch. Without `--branch`, all arguments are treated as test cases (separated by commas or spaces) and the branch defaults to `main`.

Use `--vllm-ref <ref>` to override the upstream vllm version. `<ref>` is a vllm commit SHA (full 40-char SHA recommended — GitHub cannot fetch abbreviated SHAs directly), a release tag (e.g. `v0.29.0`), or a branch name (e.g. `main`). The ref is fetched into the image's editable checkout at `/vllm-workspace/vllm` and reinstalled with `VLLM_TARGET_DEVICE=empty` (no kernel compilation), on single-node, multi-node, and accuracy test jobs alike; on multi-node it is installed in every pod. This applies to `/weekly` as well. Not specifying `--vllm-ref` keeps the vllm version baked into the nightly image, exactly as before.

> **Note**: vllm derives its version from git tags at build time. A release tag installs the exact version (e.g. `0.29.0+empty`), but a bare SHA or branch name carries no tag information, so the installed version falls back to `0.1.dev1+g<sha>` — a warning is printed in the job log when this happens. Version-gated code paths in vllm-ascend may misbehave under such a fallback version, so prefer release tags for reliable results.
>
> **Note**: vllm-ascend only tracks a specific vllm version per branch (see the `vLLM version:` line auto-maintained in the PR description). Passing an incompatible vllm ref may fail at install time or at the runtime version check; for a dev commit, set the `VLLM_VERSION` env var as described in the [versioning policy](versioning_policy.md) if needed. Also note that with `--aop_enabled`, the bisect pipeline may re-switch vllm to the version expected by each bisected commit.

Use `--aop_enabled` to enable the AOP (Aspect-Oriented Programming) pipeline, which
automatically captures test results, classifies failures (env vs. code), and triggers
binary bisect for genuine failures. By default, AOP hooks are disabled.

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
/nightly qwen3-vl-32b-instruct-w8a8 --branch releases/v0.24.0

# Run all tests on a specific branch
/nightly all --branch my-feature-branch

# Run multiple test cases (comma-separated)
/nightly test_custom_op,multi-node-deepseek-v3.2-W8A8-EP

# Run multiple test cases (space-separated, also works)
/nightly test_custom_op accuracy-group

# Run accuracy group tests (branch defaults to main)
/nightly accuracy-group

# Enable AOP bisect for all tests
/nightly all --aop_enabled

# Run specific test with AOP on a release branch
/nightly test_custom_op --branch releases/v0.24.0 --aop_enabled

# Run against a specific vllm release tag
/nightly qwen3-vl-32b-instruct-w8a8 --vllm-ref v0.29.0

# Run against a specific vllm commit (full SHA) on a release branch
/nightly test_custom_op --branch releases/v0.24.0 --vllm-ref 84030bbe3d74d99bad477a3d2e37a973ccd8865c
```

This triggers `workflow_dispatch` on both `schedule_nightly_test_a2.yaml` and `schedule_nightly_test_a3.yaml`.

> **Note**: These `schedule_*` workflows do not declare a GitHub Actions `schedule:` (cron) trigger; they are dispatched externally via `workflow_dispatch`. See [CI workflow triggers and the schedule_ prefix](../developer_guide/contribution/testing.md#ci-workflow-triggers-and-the-schedule_-prefix).

### `/cherry-pick`

Cherry-pick a PR's commits onto a specified target branch and create a new PR. This is useful for backporting fixes to release branches.

**Usage:**

| Syntax | Description |
|---|---|
| `/cherry-pick <target_branch>` | Cherry-pick onto the specified branch |

**Examples:**

```text
# Cherry-pick to a release branch
/cherry-pick releases/v0.24.0

# Cherry-pick to main
/cherry-pick main
```

A new PR will be created with the title format `[Cherry-pick] <original_title> (from #<PR_NUMBER>)` and a body linking back to the original PR.

If the cherry-pick encounters merge conflicts, the command will report the failure and the cherry-pick must be done manually.

### `/revert`

Revert a merged PR by creating a new PR that reverses its changes. The revert targets the same base branch the original PR was merged into.

**Usage:**

| Syntax | Description |
|---|---|
| `/revert` | Revert this PR (no arguments needed) |

**Example:**

```text
/revert
```

A new PR will be created with the title format `[Revert] Revert "original_title" (#PR_NUMBER)` and a body linking back to the original PR and its merge commit.

Only merged PRs can be reverted. If the revert encounters merge conflicts (e.g., because the base branch has diverged significantly), the command will report the failure and the revert must be done manually.

### `/rerun`

Re-run failed CI workflows on the current PR commit. Useful when CI jobs failed or were cancelled due to infrastructure issues.

Only jobs that did not complete successfully are re-run (failed, cancelled, timed out, or startup-failed). Jobs that already succeeded are left untouched. Runs with `cancelled` / `timed_out` / `startup_failure` conclusions are re-run per remaining job, and runs whose failure also contains cancelled jobs (e.g. a vLLM matrix leg cancelled by fail-fast) are handled the same way. Tests executed through reusable workflows (e.g. `Selected Tests`) are re-run via their caller job and are not duplicated.

**Examples:**

```text
# Re-run failed / cancelled CI jobs on this PR
/rerun
```

### `/cancel`

Force-cancel all workflow runs on the current PR commit. This cancels runs directly triggered on the PR head commit, such as the automatic E2E CI workflow (`pr_test.yaml`). Workflows triggered by slash commands (e.g., `/e2e`, `/rerun`, `/nightly`) or downstream nightly/weekly workflows are **not** affected, as those run on the `main` branch.

**Scope:**

| Cancelled | Not cancelled |
|---|---|
| `pr_test.yaml` (E2E) — automatic PR CI | `/e2e` command runs |
| `schedule_doc_getting_started_test.yaml` | `/rerun` command runs |
| `schedule_doc_linkcheck.yaml` | `/nightly` / `/weekly` command runs |
| `schedule_image_build_and_push.yaml` (if labeled) | Downstream nightly/weekly test workflows |
| `labeled_download_model_dataset.yaml` | Scheduled / `workflow_dispatch` / `push` runs |

**Examples:**

```text
# Force-cancel all CI runs on this PR
/cancel
```

> Note: This uses the `force-cancel` API endpoint, which can cancel runs even when they are in a pending or queued state waiting for runners.

## Behavior

1. When you comment a slash command, a 👀 reaction is added to your comment to indicate it has been received
2. The corresponding CI workflow is triggered asynchronously
3. Upon completion, a 🎉 reaction and a summary comment are added

## Scope

| Command | PR comments | Issue comments |
|---|---|---|
| `/e2e` | ✅ | ❌ |
| `/rerun` | ✅ | ❌ |
| `/cancel` | ✅ | ❌ |
| `/cherry-pick` | ✅ | ❌ |
| `/revert` | ✅ | ❌ |
| `/nightly` | ✅ | ❌ |

## Permission

| Command | Who can trigger |
|---|---|
| `/e2e` | PR author, or users with triage+ permission on the repository |
| `/rerun` | PR author, or users with triage+ permission on the repository |
| `/cancel` | PR author, or users with triage+ permission on the repository |
| `/cherry-pick` | PR author, or users with triage+ permission on the repository |
| `/revert` | PR author, or users with triage+ permission on the repository |
| `/nightly` | Users with triage+ permission on the repository only |

Permission is verified via the GitHub API (`repos/{owner}/{repo}/collaborators/{user}/permission`).
