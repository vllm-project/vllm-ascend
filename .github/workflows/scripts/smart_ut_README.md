# Smart UT Test Router

This document aims to introduce the usage and basic operating rules of `Smart UT` and hopes to reach a consensus with you to ensure that we can correctly place test cases according to the agreed-upon conventions when adding test cases later.
We've long been plagued by excessively long end-to-end testing durations, which is extremely detrimental to community health and developer well-being. We aim to implement selective testing based on developer pull requests (PRs):

This will be addressed from two main perspectives:

1. Intelligently selecting test cases to trigger based on developer modifications. This is easily achievable; we simply need to ensure a one-to-one correspondence between the test directory (tests/ut) and the modules in the src directory. Furthermore, for some common, fundamental modules, we will allow them to be unconditionally tested in every PR (this is meaningful because certain modules...).

2. Stateful test cases that developers are aware of. Developers only need to add a `@npu_test` decorator to specify the required NPU device type and number of chips. The system will automatically route this test case to an appropriate node for testing. I will elaborate on this point below.

## Files

| File | Role |
|------|------|
| `determine_smart_e2e_scope.py` | Main script — scans changed files, parses decorators, outputs test groups |
| `ut_config.yaml` | Maps source directories to their UT test directories |
| `ut_blacklist.yaml` | Tests excluded from running (e.g. known CPU failures) |
| `runner_label.json` | Defines available runners with chip type and NPU count |
| `tests/ut/conftest.py` | Provides the `npu_test` decorator and `RunnerDeviceType` enum |

## How It Works

```shell
PR changed files
    │
    ▼
ut_config.yaml ──► match modules ──► affected test directories
                                           │
                                           ▼
                                  AST scan @npu_test decorators
                                           │
                                           ▼
                            group by (num_npus, npu_type)
                                           │
                                           ▼
                        ut_blacklist.yaml ──► filter blacklisted tests + dedup
                                           │
                                           ▼
                        runner_label.json ──► resolve to runner label
                                           │
                                           ▼
                                  test_groups JSON output
                                           │
                                           ▼
                            GitHub Actions matrix ──► runs-on: <runner>
```

## Usage

```bash
# Route based on git diff against a base branch
python determine_smart_e2e_scope.py --diff-base origin/main

# Route based on an explicit list of changed files
python determine_smart_e2e_scope.py --changed-files vllm_ascend/ops/foo.py vllm_ascend/worker/bar.py

# Run all CPU tests regardless of module filtering (NPU tests still filtered)
python determine_smart_e2e_scope.py --diff-base origin/main --run-all-cpu

# Use a custom config file
python determine_smart_e2e_scope.py --diff-base origin/main --config path/to/ut_config.yaml
```

### Output

Written to `$GITHUB_OUTPUT` in CI, or stdout locally:

```shell
test_groups=[{"num_npus":1,"npu_type":"a2","runner":"linux-aarch64-a2b3-1","tests":"tests/ut/ops/test_layernorm.py::test_rms_norm"}]
has_tests=true
matched_modules=ops,worker
```

A human-readable summary is also printed to stderr.

## Writing Tests with `@npu_test`

### Function-level decorator

```python
from tests.ut.conftest import npu_test

@npu_test(num_npus=1, npu_type="a2")
def test_rms_norm():
    ...
```

### Class-level decorator

The decorator goes on the **class**. All methods inside share the same runner.
The output node ID is `file::ClassName` (not expanded to individual methods).

```python
from tests.ut.conftest import npu_test

@npu_test(num_npus=2, npu_type="a3")
class TestMoERouting:
    def test_topk(self):
        ...
    def test_dispatch(self):
        ...
```

### No decorator (CPU)

Tests without `@npu_test` are routed to the CPU runner.

```python
def test_config_parsing():
    ...
```

### Decorator parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_npus` | `int` | `1` | Number of NPU devices required |
| `npu_type` | `str` or `RunnerDeviceType` | `"a2"` | Chip type: `"a2"`, `"a3"`, `"310p"`, `"cpu"` |

Both forms are supported for `npu_type`:

```python
@npu_test(num_npus=1, npu_type="a2")           # string literal
@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)  # enum
```

## Output Granularity Rules

| Scenario | Output level | Example |
|----------|-------------|---------|
| All tests in directory undecorated | Directory path | `tests/ut/worker` |
| Mixed directory — decorated file | Function / class node ID | `test_foo.py::test_bar` |
| Mixed directory — undecorated file | File path | `tests/ut/ops/test_activation.py` |
| Class with `@npu_test` | Class node ID | `test_moe.py::TestMoERouting` |

## Runner Matching

Matching is **exact**: `(npu_type, num_npus)` must have a corresponding entry
in `runner_label.json`. If no match is found, the script exits with an error
listing the affected tests and available runners for that chip type.

Example error:

```shell
ERROR: The following @npu_test decorator combinations cannot be routed to any runner in runner_label.json:

  @npu_test(num_npus=1, npu_type="a3") — no runner available.
    Available a3 runners: linux-aarch64-a3-2 (a3 x2), linux-aarch64-a3-4 (a3 x4), linux-aarch64-a3-8 (a3 x8)
    Affected tests:
      - tests/ut/ops/test_foo.py::test_bar
```

## Blacklist

Tests listed in `ut_blacklist.yaml` are excluded from **all** runner groups
before the final output. This is the first-priority filter — blacklisted tests
will never appear in `test_groups`, regardless of module matching or
`--run-all-cpu`.

```yaml
# ut_blacklist.yaml
- tests/ut/worker/test_worker_v1.py
- tests/ut/kv_connector/test_remote_prefill_lifecycle.py
```

When a blacklisted file is inside a directory target (e.g. `tests/ut/kv_connector`),
the directory is automatically expanded to individual files with the blacklisted
ones removed.

## Adding a New Module

Add an entry to `ut_config.yaml`:

```yaml
- name: my_module
  optional: true
  source_file_dependencies:
    - vllm_ascend/my_module
    - tests/ut/my_module
  tests:
    - tests/ut/my_module
```

- `optional: true` — tests run only when the source files change.
- `optional: false` — tests always run on every PR (e.g. `worker`, `dummy`).

## `--run-all-cpu` Mode

When this flag is passed, **all** CPU (undecorated) tests from every module in
`ut_config.yaml` are included, bypassing module-level filtering. NPU tests are
still filtered by matched modules. This is useful during the early stage when
the module filtering mechanism is not yet mature enough for CPU tests.

Remove the flag from the workflow to restore CPU-side module filtering.
