# CI Workflow Guide

This document describes the CI workflows for `vllm-ascend`, how to add tests, and how the selective testing system works.

## Workflow Overview

| Workflow | Trigger | What it runs |
|----------|---------|---------------|
| `pr_test_light.yaml` | PR to main/dev/release branches | Lint + selective tests (UT + light E2E) |
| `pr_test_full.yaml` | PR with `ready` + `ready-for-test` labels | Selective tests (UT + full E2E) |
| `_selected_tests.yaml` | Called by `pr_test_light` / `pr_test_full` | Runs tests selected by `select_tests.py` |
| `_e2e_test.yaml` | Called by nightly/scheduled/comment-triggered workflows | Full E2E suites via `run_suite.py` |
| `_parse_trigger.yaml` | PR comment `/e2e` | Parses comment to run specific E2E tests |
| `_pre_commit.yml` | Called by `pr_test_light` | Lint and format checks |
| `schedule_nightly_test_a2.yaml` | Cron | Nightly E2E on A2 runners |
| `schedule_nightly_test_a3.yaml` | Cron | Nightly E2E on A3 runners |
| `schedule_weekly_test_a3.yaml` | Cron | Weekly E2E on A3 runners |
| `schedule_vllm_e2e_test.yaml` | Cron | E2E against current vLLM main |

## Selective Testing System

When a PR changes source files, `select_tests.py` maps changed files to affected modules in `test_config.yaml`, collects their tests, routes tests to runners, and emits a GitHub Actions matrix.

```text
PR changed files
    │
    ▼
test_config.yaml ──► resolve base inheritance ──► match modules ──► collect test paths
                                                                         │
                                                                Route by convention:
                                                                  UT:  a2/, a2_2/, a3_2/, a3_4/, 310p/
                                                                  E2E: one_card, two_card(s), four_card(s), *_310p.py
                                                                         │
                                                                runner_label.json
                                                                         │
                                                                    test_groups JSON
```

## Key Files

| File | Role |
|------|------|
| `.github/workflows/scripts/select_tests.py` | Matches changed files, scans tests, routes to runners |
| `.github/workflows/scripts/test_config.yaml` | Maps source paths to UT/E2E tests |
| `.github/workflows/scripts/runner_label.json` | Defines runner labels, chip types, NPU count, and image tags |
| `.github/workflows/scripts/config.yaml` | Full-suite E2E registry for nightly/scheduled runs |

## `test_config.yaml` Tutorial

Each module entry supports these fields:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique module name |
| `optional` | No | `true` by default. `false` means always matched when there are changed files |
| `base` | No | Module name or list of module names to inherit from |
| `source_file_dependencies` | No | Source/test paths that trigger this module |
| `exclude_source_file_dependencies` | No | Paths excluded from `source_file_dependencies` matching |
| `tests` | No | Test directories or files to run when matched |
| `skip_tests` | No | Test files to remove after directory scanning |

### Path Matching

`source_file_dependencies` and `exclude_source_file_dependencies` use the same rule:

- File path: exact match only, e.g. `vllm_ascend/attention/__init__.py`
- Directory path: matches all files below it, e.g. `vllm_ascend/attention`
- Trailing slash is ignored

Example: include all attention files except `__init__.py`:

```yaml
- name: attention_other
  optional: true
  source_file_dependencies:
    - vllm_ascend/attention
  exclude_source_file_dependencies:
    - vllm_ascend/attention/__init__.py
  tests:
    - tests/ut/attention
```

### Base Inheritance

Use `base` when a module should append another module's dependencies and tests. Inherited list fields are merged before the child fields, with duplicates removed while preserving order.

Example: `attention_gqa` inherits common attention dependencies/tests and adds GQA-specific ones:

```yaml
- name: attention_common
  optional: true
  source_file_dependencies:
    - vllm_ascend/attention/__init__.py
    - vllm_ascend/attention/attention_mask.py
    - vllm_ascend/attention/utils.py
  tests:
    - tests/ut/attention/test_attention_mask.py
    - tests/ut/attention/a2/test_common_cp.py

- name: attention_gqa
  optional: true
  base: attention_common
  source_file_dependencies:
    - vllm_ascend/attention/attention_v1.py
  tests:
    - tests/ut/attention/a2/test_attention_v1.py
```

After inheritance, `attention_gqa` behaves as if it had both `attention_common` and GQA-specific `source_file_dependencies` and `tests`.

`base` can also be a list:

```yaml
base:
  - attention_common
  - quantization
```

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
| `tests/e2e/pull_request/{light,full}/one_card/` | A2 NPU x1 |
| `tests/e2e/pull_request/{light,full}/two_card/` or `two_cards/` | A3 NPU x2 |
| `tests/e2e/pull_request/{light,full}/four_card/` or `four_cards/` | A3 NPU x4 |
| `*_310p.py` under one/two-card paths | 310P NPU x1 |
| `*_310p.py` under four-card paths | 310P NPU x4 |

### E2E Type Filtering

`--e2e-type light` keeps only `tests/e2e/pull_request/light/` paths.
`--e2e-type full` keeps only `tests/e2e/pull_request/full/` paths.
Non-`pull_request` E2E paths are always included.

## Adding a New UT Test

1. Put the test in the right directory:

   - CPU: `tests/ut/<module>/test_foo.py`
   - A2 x1: `tests/ut/<module>/a2/test_foo.py`
   - A2 x2: `tests/ut/<module>/a2_2/test_foo.py`
   - A3 x2: `tests/ut/<module>/a3_2/test_foo.py`
   - A3 x4: `tests/ut/<module>/a3_4/test_foo.py`
   - 310P x1: `tests/ut/<module>/310p/test_foo.py`

2. Add or update the matching module in `test_config.yaml`:

```yaml
- name: my_module
  optional: true
  source_file_dependencies:
    - vllm_ascend/my_module
  tests:
    - tests/ut/my_module
```

If `tests` points to a directory, `select_tests.py` scans `test_*.py` files and routes NPU subdirectories automatically.

## Adding a New E2E Test

1. Put the test under the correct PR directory:

   - Light 1-card: `tests/e2e/pull_request/light/one_card/test_new_feature.py`
   - Full 1-card: `tests/e2e/pull_request/full/one_card/test_new_feature.py`
   - Full 2-card: `tests/e2e/pull_request/full/two_cards/test_new_feature.py`
   - Full 4-card: `tests/e2e/pull_request/full/four_cards/test_new_feature.py`

2. Register it in `.github/workflows/scripts/test_config.yaml` for PR selective testing.
3. Register it in `.github/workflows/scripts/config.yaml` for nightly/scheduled full-suite testing.

Example:

```yaml
- name: e2e_my_feature
  optional: true
  source_file_dependencies:
    - vllm_ascend/my_feature
    - tests/e2e/pull_request/full/one_card/test_my_feature.py
  tests:
    - tests/e2e/pull_request/light/one_card/test_my_feature_light.py
    - tests/e2e/pull_request/full/one_card/test_my_feature.py
    - tests/e2e/pull_request/full/two_cards/test_my_feature_distributed.py
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

## Testing Changes to `select_tests.py`

```bash
PYTHONPATH=.github/workflows/scripts pytest -sv .github/workflows/scripts/test_select_tests.py
ruff check .github/workflows/scripts/select_tests.py .github/workflows/scripts/test_select_tests.py
bash format.sh ci
```

## Full-Suite E2E Partitioning

Nightly and scheduled E2E suites are partitioned by `run_suite.py` using `estimated_time` from `config.yaml`.

```bash
# Run a suite locally
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard

# Simulate partitioned execution
python3 .github/workflows/scripts/run_suite.py --suite e2e-singlecard --auto-partition-id 0 --auto-partition-size 2
```
