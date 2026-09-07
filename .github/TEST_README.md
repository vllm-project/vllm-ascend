# CI Workflow Guide

This document describes the CI workflows for `vllm-ascend`, how to add tests, and how the selective testing system works.

## Workflow Overview

| Workflow | Trigger | What it runs |
|----------|---------|---------------|
| `pr_test.yaml` | PR to main/dev/release branches | Lint + selective tests (UT + E2E) |
| `_selected_tests.yaml` | Called by `pr_test.yaml` | Runs tests selected by `select_tests.py` |
| `_parse_trigger.yaml` | PR comment `/e2e` | Parses comment to run specific E2E tests |
| `_pre_commit.yml` | Called by `pr_test.yaml` | Lint and format checks |
| `schedule_nightly_test_a2.yaml` | Cron | Nightly E2E on A2 runners |
| `schedule_nightly_test_a3.yaml` | Cron | Nightly E2E on A3 runners |
| `schedule_weekly_test_a3.yaml` | Cron | Weekly E2E on A3 runners |

## Selective Testing System

When a PR changes source files, the precision-testing pipeline
(`test_selector.py`) recommends tests from historical coverage data and the
PR diff, then `select_tests.py` routes the recommended tests to runners and
emits a GitHub Actions matrix.

```text
PR changed files
    │
    ▼
test_selector.py (coverage + AST) ──► recommended pytest targets
                                         │
                              select_tests.py --test-list-file
                                         │
                            runner_mapping regex routing
                                         │
                              default partition
                                         │
                            pinned_routes (optional)
                                         │
                            partition runner_label
                                         │
                              runner_label.json
                                         │
                                  test_groups JSON
```

## Key Files

| File | Role |
|------|------|
| `.github/workflows/scripts/test_selector.py` | Recommends tests from coverage data and the PR diff (AST based) |
| `.github/workflows/scripts/select_tests.py` | Routes test targets to runners and emits the matrix |
| `.github/workflows/scripts/test_config.yaml` | Routing metadata: curated suites, skip list, runner mapping, partitions, estimated times |
| `.github/workflows/scripts/runner_label.json` | Defines runner labels, chip types, NPU count, and image tags |

`runner_mapping` maps test paths to default logical partitions such as
`a3-2` or `a3-4`. After selection, skip filtering, and containment-aware
deduplication, `pinned_routes` can move selected files into dedicated logical
partitions without changing their technical directory structure. A file-level
pin also applies when only one of its `::nodeid` targets is selected. If both a
bare file and one of its nodeids are selected, the bare file wins to prevent
duplicate execution. Each entry in `partition` selects an exact key from
`runner_label.json` and sets the load-balanced group count.
Logical partition names use display-ready labels such as `a3-2` and
`a3-800i-2`; GitHub jobs render them as `a3-2 card-(part 1-3)` while the
numeric `partition` value remains available for artifacts and load balancing.

Workflows that temporarily reroute selected tests can use
`--runner-label-override` with an exact label from `runner_label.json`. This
override is applied after pinned routes, has the highest priority, and creates
a transient single-shard partition for that invocation.

## `select_tests.py` Modes

| Mode | Flag | Used by |
|------|------|---------|
| Recommended | `--test-list-file <file>` | PR precision testing (coverage/AST recommendation) |
| Explicit | `--explicit-e2e-tests <path>...` | `/e2e` PR comment command |
| Full suite | `--all-tests` | `ready-all` PRs, scheduled full scans |
| Curated | `--curated <name>` | Curated suites (e.g. A5) |

`test_config.yaml` holds the routing metadata consumed by all modes:

| Field | Description |
|-------|-------------|
| `curated_tests` | Named test lists selected via `--curated <name>` |
| `skip_tests` | Test files removed from any selection after scanning |
| `runner_mapping` | Regex patterns mapping test paths to logical partitions |
| `estimated_times` | Per-test estimated seconds for load-balanced partitioning |
| `pinned_routes` | Files moved to dedicated partitions after selection |
| `partition` | Logical partition → runner label + load-balanced group count |

## Runner Routing

### UT Routing

No decorator is needed. UT runner routing is determined by path:

| Directory pattern | Runner |
|-------------------|--------|
| `tests/ut/<module>/` | CPU |
| `tests/ut/<module>/a2/` | A2 NPU x1 |
| `tests/ut/<module>/a2_2/` | A2 NPU x2 |
| `tests/ut/<module>/a3_2/` | A3 NPU x2 |
| `tests/ut/<module>/a3_4/` | A3 NPU x4 |
| `tests/ut/<module>/310p/` | 310P NPU x1 |

`tests/ut/_310p/` is intentionally not treated as `310p/`; it runs on CPU in mock mode.

### E2E Routing

All E2E tests run on NPU. E2E routing is determined by directory or `_310p` filename suffix:

| Pattern | Runner |
|---------|--------|
| `tests/e2e/pull_request/one_card/` | A2 NPU x1 |
| `tests/e2e/pull_request/two_card/` | A3 NPU x2 |
| `tests/e2e/pull_request/four_card/` | A3 NPU x4 |
| `tests/e2e/pull_request/eight_card/` | A3 NPU x8 |
| `*_310p.py` under one/two-card paths | 310P NPU x1 |
| `*_310p.py` under four-card paths | 310P NPU x4 |

## Adding a New UT Test

1. Put the test in the right directory:

   - CPU: `tests/ut/<module>/test_foo.py`
   - A2 x1: `tests/ut/<module>/a2/test_foo.py`
   - A2 x2: `tests/ut/<module>/a2_2/test_foo.py`
   - A3 x2: `tests/ut/<module>/a3_2/test_foo.py`
   - A3 x4: `tests/ut/<module>/a3_4/test_foo.py`
   - 310P x1: `tests/ut/<module>/310p/test_foo.py`

The directory determines the runner. No configuration change is needed: the
precision-testing recommendation and `--all-tests` mode pick the file up
automatically from the test tree.

## Adding a New E2E Test

1. Put the test under the correct card directory:

   - 1-card: `tests/e2e/pull_request/one_card/test_new_feature.py`
   - 2-card: `tests/e2e/pull_request/two_card/test_new_feature.py`
   - 4-card: `tests/e2e/pull_request/four_card/test_new_feature.py`
   - 8-card: `tests/e2e/pull_request/eight_card/test_new_feature.py`

The directory determines the runner. No configuration change is needed.

## Adding a Curated Suite

Add a named list to `curated_tests` in `.github/workflows/scripts/test_config.yaml`:

```yaml
curated_tests:
  a5:
    - tests/e2e/pull_request/four_card/test_data_parallel_tp2.py
```

Then run it with `select_tests.py --curated a5` (optionally combined with
`--runner-label-override`).

## Running Selective Tests Locally

```bash
# Route based on a recommended test list (mirrors the PR precision flow)
printf 'tests/ut/test_envs.py\ntests/e2e/pull_request/one_card/test_foo.py\n' > /tmp/recommended.txt
python3 .github/workflows/scripts/select_tests.py --test-list-file /tmp/recommended.txt

# Run a specific subset of e2e tests (mirrors the /e2e slash command)
python3 .github/workflows/scripts/select_tests.py \
  --explicit-e2e-tests tests/e2e/pull_request/one_card/test_foo.py \
                        tests/e2e/pull_request/two_card/test_bar.py

# Run a single test method (supports the same ::nodeid syntax as pytest)
python3 .github/workflows/scripts/select_tests.py \
  --explicit-e2e-tests \
    tests/e2e/pull_request/one_card/test_foo.py::TestClass::test_method

# Run the full suite (mirrors ready-all PRs and scheduled scans)
python3 .github/workflows/scripts/select_tests.py --all-tests

# Run a curated suite
python3 .github/workflows/scripts/select_tests.py --curated a5
```

## Testing Changes to `select_tests.py`

```bash
PYTHONPATH=.github/workflows/scripts pytest -sv .github/workflows/scripts/test_select_tests.py
ruff check .github/workflows/scripts/select_tests.py .github/workflows/scripts/test_select_tests.py
bash format.sh ci
```
