# CI Workflow Guide

This document describes the CI workflows for `vllm-ascend`, how to add new test cases, and how the selective testing system works.

## Workflow Overview

| Workflow | Trigger | What it runs |
|----------|---------|---------------|
| `pr_test_light.yaml` | PR to main/dev/release branches | Lint + selective tests (UT + light E2E) |
| `pr_test_full.yaml` | PR with `ready` + `ready-for-test` labels | Selective tests (UT + full E2E) |
| `_selected_tests.yaml` | Called by `pr_test_light` / `pr_test_full` | Runs the tests selected by `select_tests.py` |
| `_e2e_test.yaml` | Called by nightly/scheduled/comment-triggered workflows | Full E2E suites via `run_suite.py` |
| `_parse_trigger.yaml` | PR comment `/e2e` | Parses comment to run specific E2E tests |
| `_pre_commit.yml` | Called by `pr_test_light` | Lint and format checks |
| `schedule_nightly_test_a2.yaml` | Cron | Nightly E2E on A2 runners |
| `schedule_nightly_test_a3.yaml` | Cron | Nightly E2E on A3 runners |
| `schedule_weekly_test_a3.yaml` | Cron | Weekly E2E on A3 runners |
| `schedule_vllm_e2e_test.yaml` | Cron | E2E against current vLLM main |

## Selective Testing System

When a PR changes source files, only the affected tests should run — not the entire suite. This is handled by `select_tests.py` and configured in `test_config.yaml`.

### How It Works

```text
PR changed files
    │
    ▼
test_config.yaml ──► match modules ──► affected test paths
                                            │
                                   Route by directory convention:
                                     UT:  a2/, a3_2/, 310p/ → NPU runner
                                          (no convention)   → CPU runner
                                     E2E: 1-card, 2-card, 4-card, 310p
                                            │
                                   runner_label.json ──► resolve runner
                                            │
                                       test_groups JSON
                                            │
                              GitHub Actions matrix ──► runs-on: <runner>
```

### Key Files

| File | Role |
|------|------|
| `select_tests.py` | Matches changed files to test paths, routes by convention |
| `test_config.yaml` | Maps source directories to test directories (UT + E2E) |
| `runner_label.json` | Defines available runners with chip type, NPU count, and image tag |
| `selective_test_README.md` | Detailed reference for the selective testing system |

### UT Runner Routing (by directory convention)

No decorator is needed. The path of a test file determines the runner:

| Directory pattern | Runner |
|-------------------|--------|
| `tests/ut/<module>/` | CPU |
| `tests/ut/<module>/a2/` | A2 NPU x1 |
| `tests/ut/<module>/a2_2/` | A2 NPU x2 |
| `tests/ut/<module>/a3_2/` | A3 NPU x2 |
| `tests/ut/<module>/a3_4/` | A3 NPU x4 |
| `tests/ut/<module>/310p/` | 310P NPU x1 |

### E2E Runner Routing (by directory convention)

All E2E tests run on NPU. Routing is determined by the card-count directory:

| Directory pattern | Runner |
|-------------------|--------|
| `tests/e2e/pull_request/{light,full}/1-card/` | A2 NPU x1 |
| `tests/e2e/pull_request/{light,full}/2-card[s]/` | A3 NPU x2 |
| `tests/e2e/pull_request/{light,full}/4-card[s]/` | A3 NPU x4 |
| `tests/e2e/310p/singlecard/` | 310P NPU x1 |
| `tests/e2e/310p/multicard/` | 310P NPU x4 |

### E2E Type Filtering

`select_tests.py` accepts `--e2e-type light` or `--e2e-type full` to limit E2E tests to the corresponding `pull_request` subdirectory:

- `pr_test_light.yaml` uses `--e2e-type light`
- `pr_test_full.yaml` uses `--e2e-type full`

Non-`pull_request` E2E paths (e.g. `tests/e2e/310p/`) are always included regardless of the filter.

## Adding a New UT Test Case

1. **Write the test**: Place the `.py` file in the appropriate directory:
   - CPU tests: `tests/ut/<module>/test_foo.py`
   - A2 NPU tests: `tests/ut/<module>/a2/test_foo.py`
   - A3 NPU x2 tests: `tests/ut/<module>/a3-2/test_foo.py`

2. **Update `test_config.yaml`**: Ensure the module entry exists with the correct `source_file_dependencies`. If the module already has an entry (e.g. `ops`), no change is needed — new test files are automatically discovered.

Example — adding a new module:

```yaml
- name: my_module
  optional: true
  source_file_dependencies:
    - vllm_ascend/my_module
    - tests/ut/my_module
  tests:
    - tests/ut/my_module
```

## Adding a New E2E Test Case

1. **Write the test**: Place the `.py` file in the appropriate directory under `tests/e2e/pull_request/`:
   - `tests/e2e/pull_request/light/1-card/test_new_feature.py` — light smoke test
   - `tests/e2e/pull_request/full/1-card/test_new_feature.py` — full 1-card test
   - `tests/e2e/pull_request/full/2-cards/test_new_feature.py` — full 2-card test

2. **Add to `config.yaml`**: For full-suite runs (nightly, scheduled), add an entry with `name` and `estimated_time` to `.github/workflows/scripts/config.yaml` under the corresponding suite.

3. **Add to `test_config.yaml`**: For selective test runs (PR-triggered), add the test file to the appropriate E2E module entry so that changes to the relevant source code will trigger it.

Example — adding an E2E test triggered by `vllm_ascend/my_feature` changes:

```yaml
- name: e2e_my_feature
  optional: true
  source_file_dependencies:
    - vllm_ascend/my_feature
    - tests/e2e/pull_request   # so editing the test file itself also triggers it
  tests:
    - tests/e2e/pull_request/light/1-card/test_my_feature_light.py
    - tests/e2e/pull_request/full/1-card/test_my_feature.py
    - tests/e2e/pull_request/full/2-cards/test_my_feature_distributed.py
```

### E2E Test Files: Two Registries

Each E2E test file must be registered in **both** places:

| Registry | File | Purpose |
|----------|------|---------|
| Selective testing | `test_config.yaml` | Determines which E2E tests run on a PR |
| Full-suite testing | `config.yaml` | Determines suites, estimated times, and partitioning for nightly/scheduled runs |

If you add an E2E test but forget `test_config.yaml`, it will not run on PRs. If you forget `config.yaml`, it will not run in nightly/scheduled suites.

## Automatic Partitioning

For full-suite runs (nightly, scheduled, `e2e-full`), tests are partitioned across parallel jobs by `run_suite.py` using estimated times for load balancing.

### Principle

1. **Read Configuration**: Read all non-skipped test cases and their `estimated_time` from `config.yaml`.
2. **Sort (Balanced Assignment)**: Sort by `estimated_time` descending — heaviest tasks first.
3. **Assign**: Each case goes to the partition with the current minimum total time (greedy).
4. **Re-sort (Fast Feedback)**: Within each partition, sort ascending by `estimated_time` so quick tests run first.

### Running Suites Locally

```bash
# Run the full e2e-singlecard suite
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard

# Simulate partitioned execution (partition 0 of 2)
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard --auto-partition-id 0 --auto-partition-size 2
```

## Running Selective Tests Locally

```bash
# Route based on git diff
python3 .github/workflows/scripts/select_tests.py --diff-base origin/main

# Route based on explicit changed files
python3 .github/workflows/scripts/select_tests.py --changed-files vllm_ascend/ops/foo.py

# Light E2E only
python3 .github/workflows/scripts/select_tests.py --diff-base origin/main --e2e-type light

# Full E2E only
python3 .github/workflows/scripts/select_tests.py --diff-base origin/main --e2e-type full

# Run all CPU UT tests regardless of module matching
python3 .github/workflows/scripts/select_tests.py --diff-base origin/main --run-all-cpu
```
