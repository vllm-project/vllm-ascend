# E2E CI Test

This document explains how to trigger specific E2E tests against your PR code via a
comment command, without running the full E2E test suite.

## Background

The `E2E-Full` workflow (`pr_test.yaml`) normally runs the complete E2E test suite
when a PR has `ready` label. This is expensive in CI resources
and time.

Authorized users can trigger only the specific test files they care about by posting a
`/e2e` comment on the PR, then adding the `ready` label.

## How to Trigger

### 1. Post a comment

First, post a comment on the PR specifying which test paths to run:

```text
/e2e [test-path-1] [test-path-2] ...
```

- Each path must be a valid pytest path relative to the repository root.
- Multiple paths can be listed in a single comment, separated by spaces.
- A specific test case can be targeted using `::` notation.

| Comment format | Effect |
|---|---|
| `/e2e tests/e2e/pull_request/one_card/test_foo.py` | Run one test file on one_card |
| `/e2e tests/e2e/pull_request/two_card/test_bar.py` | Run one test file on two_card |
| `/e2e path1 path2 path3` | Run multiple files, routed by path pattern |
| `/e2e tests/e2e/pull_request/one_card/test_foo.py::test_case` | Run a specific test case |

### 2. Add the label

After posting the comment, add the **`ready`** label to your PR.
Adding the label is what actually **triggers** the workflow — at that point the workflow
reads the existing comments to find the `/e2e` command.

!!! note

    Only repository **Contributors** (Triage role) and **Maintainers** (Write role) can add
    labels. If you do not have this permission, ask a maintainer to add the label for you.
    You can find the list of maintainers and contributors by checking the
    [CODEOWNERS](https://github.com/vllm-project/vllm-ascend/blob/main/.github/CODEOWNERS)
    file.

!!! warning

    The comment must be posted **before** the label is added. If you add the label first,
    the workflow will find no `/e2e` comment and will not trigger any per-test runs.

!!! note

    Additionally, only the **PR author** or collaborators with **write or admin** repository
    access can trigger tests via comment. The workflow validates the commenter's permission
    before proceeding.

### 3. Wait for results

GitHub Actions will trigger the `E2E-Full` workflow. Only the hardware jobs matching
the provided test paths will run, which saves CI resources.

## Path Routing Rules

The workflow automatically routes each test path to the correct hardware runner based
on path patterns:

| Path pattern | Hardware | Runner |
|---|---|---|
| `two_card` in path | two_card A3 NPU | `linux-aarch64-a3-2` |
| `four_card` in path | four_card A3 NPU | `linux-aarch64-a3-4` |
| `_310p` in filename under one/two_card | Ascend 310P x1 | `linux-aarch64-310p-*` |
| `_310p` in filename under four_card | Ascend 310P x4 | `linux-aarch64-310p-*` |
| All other paths | one_card A2 NPU | `linux-aarch64-a2b3-1` |

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

310P tests use `_310p` subdirectories or `_310p.py` filename suffix under the
corresponding card directory:

```text
tests/e2e/pull_request/one_card/_310p/   # 310P single card
tests/e2e/pull_request/four_card/_310p/  # 310P four card
```

## Comparison with Full E2E Suite

| Aspect | Full E2E suite | Per-test comment trigger |
|---|---|---|
| Trigger | `ready` labels | `/e2e` comment + `ready` label |
| Scope | All E2E tests | Only specified test paths |
| Who can trigger | Anyone who can add labels | PR author or write/admin collaborator |
| Use case | Pre-merge validation | Iterative debugging of specific tests |

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

Re-trigger after fixing an issue: just push a new commit. The `synchronize` event
re-runs the workflow and picks up the existing `/e2e` comment automatically — no need
to post a new comment.

## Troubleshooting

**The workflow did not start after I added the label.**

- Make sure the `/e2e` comment was posted **before** the label was added.
  If the label was added first, remove it and re-add it after posting the comment.
- Check that the comment starts exactly with `/e2e` followed by at least one path,
  with no leading spaces or extra characters before the slash.
- To re-trigger after fixing an issue, simply push a new commit — the workflow will
  reuse the existing `/e2e` comment automatically.

**Tests ran on the wrong hardware.**

- Check that the path includes the expected directory segment (`one_card`, `two_card`,
  `four_card`, or `_310p`). Paths that do not match any of these patterns are routed to
  the one_card runner by default.

**The `parse-comment` job skipped with a permission error.**

- Only the PR author or write/admin collaborators can use the comment trigger.
  Ask a maintainer to post the `/e2e` comment instead.
