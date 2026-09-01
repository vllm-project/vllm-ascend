# E2E CI Test

This document explains how to trigger specific E2E tests against your PR code via a
comment command, without running the PR-selected test suite.

## Background

The `E2E` workflow ([`pr_test.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/pr_test.yaml)) runs a PR-selected set of tests when a PR has one of the `ready-precise`, `ready-all`, or `ready-a5` labels.
This is expensive in CI resources and time.

Authorized users can trigger only the specific test files they care about by posting a
`/e2e` comment on the PR.

## How to Trigger

### 1. Post a comment

Post a comment on the PR specifying which test paths to run:

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

The comment itself triggers the workflow — no label is required.

### 2. Wait for results

GitHub Actions will trigger the `Handle /e2e Command` workflow. Only the hardware jobs matching
the provided test paths will run, which saves CI resources.

## Path Routing Rules

The workflow routes each test path to the correct hardware runner via the
`runner_mapping` regex patterns in
[`.github/workflows/scripts/test_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/scripts/test_config.yaml). Each pattern maps
to a logical partition, which selects an exact runner label from `runner_label.json`:

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

## Comparison with PR-Selected Tests

| Aspect | PR-selected tests (`ready-precise` / `ready-all`) | Per-test comment trigger |
|---|---|---|
| Trigger | Label | `/e2e` comment |
| Scope | Tests recommended by the precision-testing pipeline (coverage + AST based), or the full suite under `ready-all` | Only specified test paths |
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

**The workflow did not start after I posted the comment.**

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
