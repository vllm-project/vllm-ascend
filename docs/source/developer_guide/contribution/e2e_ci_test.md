# E2E CI Test

This document explains how to trigger specific E2E tests against your PR code via a
comment command, without running the full E2E test suite.

## Background

The `E2E-Full` workflow (`pr_test_full.yaml`) normally runs the complete E2E test suite
when a PR has both `ready` and `ready-for-test` labels. This is expensive in CI resources
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
| `/e2e tests/e2e/singlecard/test_foo.py` | Run one test file on singlecard |
| `/e2e tests/e2e/multicard/2-cards/test_bar.py` | Run one test file on 2-card |
| `/e2e path1 path2 path3` | Run multiple files, routed by path pattern |
| `/e2e tests/e2e/singlecard/test_foo.py::test_case` | Run a specific test case |

### 2. Add the label

After posting the comment, add the **`ready`** label to your PR.
Adding the label is what actually **triggers** the workflow — at that point the workflow
reads the existing comments to find the `/e2e` command.

:::{note}
Only repository **Contributors** (Triage role) and **Maintainers** (Write role) can add
labels. If you do not have this permission, ask a maintainer to add the label for you.
You can find the list of maintainers and contributors by checking the
[CODEOWNERS](https://github.com/vllm-project/vllm-ascend/blob/main/.github/CODEOWNERS)
file.
:::

:::{important}
The comment must be posted **before** the label is added. If you add the label first,
the workflow will find no `/e2e` comment and will not trigger any per-test runs.
:::

:::{note}
Additionally, only the **PR author** or collaborators with **write or admin** repository
access can trigger tests via comment. The workflow validates the commenter's permission
before proceeding.
:::

### 3. Wait for results

GitHub Actions will trigger the `E2E-Full` workflow. Only the hardware jobs matching
the provided test paths will run, which saves CI resources.

## Path Routing Rules

The workflow automatically routes each test path to the correct hardware runner based
on path patterns:

| Path pattern | Hardware | Runner |
|---|---|---|
| `multicard/2-cards` in path | 2-card A3 NPU | `linux-aarch64-a3-2` |
| `multicard/4-cards` in path | 4-card A3 NPU | `linux-aarch64-a3-4` |
| `310p` in path | Ascend 310P | `linux-aarch64-310p-*` |
| All other paths | Singlecard A2 NPU | `linux-aarch64-a2b3-1` |

When paths from multiple categories are listed in a single comment, each category's
tests run on its respective hardware in parallel.

## Test Path Reference

The `tests/e2e/` directory is organized by hardware category:

```text
tests/e2e/
├── singlecard/          # Single A2 card tests → singlecard runner
├── multicard/
│   ├── 2-cards/         # 2-card tests → 2-card runner
│   └── 4-cards/         # 4-card tests → 4-card runner
└── 310p/                # Ascend 310P tests → 310P runner
    ├── singlecard/
    └── multicard/
```

## Comparison with Full E2E Suite

| | Full E2E suite | Per-test comment trigger |
|---|---|---|
| Trigger | `ready` + `ready-for-test` labels | `/e2e` comment + `ready` label |
| Scope | All E2E tests | Only specified test paths |
| Who can trigger | Anyone who can add labels | PR author or write/admin collaborator |
| Use case | Pre-merge validation | Iterative debugging of specific tests |

## Examples

Run a single singlecard test:

```text
/e2e tests/e2e/singlecard/test_offline_inference.py
```

Run a 2-card test:

```text
/e2e tests/e2e/multicard/2-cards/test_quantization.py
```

Run tests across multiple hardware categories in one comment:

```text
/e2e tests/e2e/singlecard/test_offline_inference.py tests/e2e/multicard/2-cards/test_quantization.py
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

- Check that the path includes the expected directory segment (`2-cards`, `4-cards`,
  or `310p`). Paths that do not match any of these patterns are routed to the
  singlecard runner by default.

**The `parse-comment` job skipped with a permission error.**

- Only the PR author or write/admin collaborators can use the comment trigger.
  Ask a maintainer to post the `/e2e` comment instead.
