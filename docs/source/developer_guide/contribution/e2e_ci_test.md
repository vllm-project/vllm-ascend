# E2E CI Test

This document explains how to trigger specific E2E tests against your PR code via a
slash command comment, without running the full E2E test suite.

## Background

E2E tests are expensive in CI resources and time. Authorized users can trigger only the
specific test files they care about by posting a `/e2e` comment on the PR.

## How to Trigger

### 1. Post a comment

Post a comment on the PR with the test paths to run:

```text
/e2e <test-path> [test-path-2 ...] [--vllm <version1,version2>]
```

- Each path must be under `tests/e2e/pull_request/` and include a subdirectory
  (e.g. `tests/e2e/pull_request/one_card/test_foo.py`).
- Multiple paths can be listed in a single comment, separated by spaces.
- Use `--vllm <versions>` to specify which vLLM versions to test against
  (comma-separated commit SHA or release tag). Without `--vllm`, tests run against
  both the pinned main commit and the latest release tag by default.

| Comment format | Effect |
|---|---|
| `/e2e tests/e2e/pull_request/one_card/test_foo.py` | Run one test file on one_card |
| `/e2e tests/e2e/pull_request/two_card/test_bar.py` | Run one test file on two_card |
| `/e2e path1 path2 path3` | Run multiple files, routed by path pattern |
| `/e2e tests/e2e/pull_request/one_card/test_foo.py --vllm <release-tag>` | Run against a specific vLLM version |

### 2. Wait for results

Once the comment is posted, the workflow is triggered automatically. A 🚀 reaction is
added to your comment to confirm receipt.

Upon completion, the bot updates the comment with one of the following outcomes:

- **completed successfully** — all tests passed
- **found no tests to run** — no tests matched the provided paths; check the test paths
- **was cancelled** — the workflow was cancelled
- **failed** — one or more tests failed

:::{note}
Only the **PR author** or collaborators with **triage or higher** repository permission
can trigger `/e2e`. If you lack permission, ask a maintainer to post the comment instead.
:::

## Path Routing Rules

The workflow automatically routes each test path to the correct hardware runner based
on path patterns:

| Test path contains | Runner |
|---|---|
| `four_card/_310p` | 310P 4-card |
| `_310p` (under `one_card`/`two_card`) | 310P single card |
| `four_card` | A3 4-card |
| `two_card` | A3 2-card |
| Others (e.g. `one_card`) | A2 single card |

When paths from multiple categories are listed in a single comment, each category's
tests run on its respective hardware in parallel.

## Test Path Reference

The `tests/e2e/pull_request/` directory is organized by hardware category:

```text
tests/e2e/pull_request/
├── one_card/          # Single card tests → A2 NPU x1 runner
├── two_card/          # Two card tests → A3 NPU x2 runner
├── four_card/         # Four card tests → A3 NPU x4 runner
```

310P tests use `_310p` subdirectories under the corresponding card directory:

```text
tests/e2e/pull_request/one_card/_310p/   # 310P single card
tests/e2e/pull_request/four_card/_310p/  # 310P four card
```

## Examples

Run a single one_card test:

```text
/e2e tests/e2e/pull_request/one_card/test_offline_inference.py
```

Run a two_card test:

```text
/e2e tests/e2e/pull_request/two_card/test_data_parallel.py
```

Run tests across multiple hardware categories in one comment:

```text
/e2e tests/e2e/pull_request/one_card/test_offline_inference.py tests/e2e/pull_request/two_card/test_data_parallel.py
```

Run tests against a specific vLLM version:

```text
/e2e tests/e2e/pull_request/one_card/test_offline_inference.py --vllm <release-tag>
```

## Troubleshooting

**The workflow did not start after I posted the comment.**

- Check that the comment starts exactly with `/e2e` followed by at least one valid path,
  with no leading spaces or extra characters before the slash.
- Confirm you are the PR author or have triage+ repository permission.

**The bot replied "found no tests to run".**

- Verify that each test path is under `tests/e2e/pull_request/<subdir>/`.
  Paths outside this prefix are rejected.

**Tests ran on the wrong hardware.**

- Check that the path includes the expected directory segment (`one_card`, `two_card`,
  `four_card`, or `_310p`). Paths that do not match any of these patterns are routed to
  the one_card runner by default.

**The bot replied with a permission error.**

- Only the PR author or users with triage+ permission can trigger `/e2e`.
  Ask a maintainer to post the comment instead.
