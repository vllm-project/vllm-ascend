#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Determine which tests to run based on changed files in a PR.

Two input modes are supported (mutually exclusive):

- ``--changed-files`` / ``--diff-base``: PR-driven. The input is a list of
  source files changed in a PR. Modules are matched by their
  ``source_file_dependencies``, and their configured tests are collected
  and routed to runners.

- ``--explicit-e2e-tests``: Slash-command driven. The input is a list of
  e2e test paths (files or directories) supplied via the ``/e2e`` PR
  comment. Module matching is bypassed entirely; each path is routed
  directly to the appropriate runner.

- ``--test-list-file``: File-driven. The input is a text file with one
  pytest target per line (UT or E2E). Lines starting with ``#`` and
  blank lines are ignored. Supports file paths, directories, and
  ``::nodeid`` suffixes for test classes or methods.

Pipeline (PR-driven mode):
  1. Diff       -- get changed files from git.
  2. Match      -- identify affected modules via test_config.yaml.
  3. Collect    -- gather test paths (always resolved to individual files).
  4. Route      -- map tests to logical partitions via runner_mapping.
  5. Pin        -- move configured files to dedicated logical partitions.
  6. Partition  -- select exact runner labels and split groups by estimated time.
  7. Output     -- write test_groups / has_tests / matched_modules.

Test/tool-scoped optimization:
  In PR-driven mode, the workflow passes both the full diff and the changed
  files matched by the PR test path filter. If the filtered files contain only
  test paths and/or non-bisect ``tools`` paths, normal module matching is
  bypassed. The selector runs ``default_cpu_ut`` and directly changed test
  files. Files outside the PR test filter, such as nightly-only files and docs,
  do not widen the test scope. Bisect changes retain their dedicated policy.

Bisect-tool optimization:
  If a PR is scoped to ``tools/bisect`` and its paired UT/config/format files,
  the always-on modules are skipped and only modules whose dependencies match
  the changed files are selected. This keeps maintenance-only tool changes
  from triggering the full CPU and NPU regression suite.

Routing is driven by ``test_config.yaml`` ``runner_mapping:`` (regex patterns).
Each entry in ``partition:`` selects an exact label from runner_label.json and
defines the number of load-balanced groups.
See ``test_config.yaml`` for details.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import regex as re
import yaml

_SCRIPT_DIR = Path(__file__).parent
_CONFIG_PATH = _SCRIPT_DIR / "test_config.yaml"
_RUNNER_LABEL_PATH = _SCRIPT_DIR / "runner_label.json"


class NpuType(str, Enum):
    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    A5 = "a5"
    CPU = "cpu"


@dataclass(frozen=True)
class RunnerInfo:
    num_npus: int
    npu_type: NpuType
    label: str
    image_tag: str = ""
    csrc_cache_target: str = ""


@dataclass(frozen=True)
class PartitionInfo:
    runner_label: str
    count: int


PartitionKey = str
_DEFAULT_KEY: PartitionKey = "cpu-0"

# The always-on CPU UT module. In test-only changes, only this module
# is selected for UT runs (along with the changed test files).
DEFAULT_CPU_UT_MODULE = "default_cpu_ut"

# Coverage-based recommendation emits this batch label (not a real pytest
# path) to represent the always-on CPU UT suite. Map it to ``tests/ut`` so
# the ``--test-list-file`` flow runs the same CPU UTs as the diff flow.
CPU_UT_BATCH_ALIAS = "cpu-ut"
CPU_UT_BATCH_PATH = "tests/ut"

_BISECT_TOOL_ROOTS = ("tools/bisect", "tests/ut/tools/bisect")
_BISECT_TOOL_SUPPORT_FILES = {
    ".github/workflows/scripts/select_tests.py",
    ".github/workflows/scripts/test_config.yaml",
    ".github/workflows/scripts/test_select_tests.py",
    "csrc/build.sh",
    "tests/ut/tools/__init__.py",
}

# Populated by _load_runner_mapping(). Ordered list of
# (regex, {variant: logical partition key}).
_RUNNER_MAPPING: list[tuple[re.Pattern, dict[str, PartitionKey]]] = []


def _load_runner_mapping(meta: dict) -> None:
    """Load runner mapping from the config meta dict into ``_RUNNER_MAPPING``.

    Config format::

        runner_mapping:
          <regex_pattern>:
            default: <partition_key>
            "310p": <partition_key>   # optional override for 310P files

    Patterns are sorted longest first so more specific patterns match first.
    """
    global _RUNNER_MAPPING
    _RUNNER_MAPPING = []
    raw = list((meta.get("runner_mapping", {}) or {}).items())
    raw.sort(key=lambda x: -len(x[0]))
    for pattern_str, runner_config in raw:
        runners: dict[str, PartitionKey] = {}
        for key, val in runner_config.items():
            runners[key] = str(val)
        _RUNNER_MAPPING.append((re.compile(pattern_str), runners))


def _resolve_partition(file_path: str) -> PartitionKey | None:
    """Match *file_path* against ``_RUNNER_MAPPING``.

    Returns the ``default`` logical partition for the first matching pattern.
    If the filename contains ``_310p`` and the matched pattern has
    a ``"310p"`` entry, that entry is returned instead.
    """
    route_path = _as_posix_path(_pytest_node_file_path(file_path))
    for pattern, runners in _RUNNER_MAPPING:
        if pattern.search(route_path):
            if "_310p" in Path(route_path).name and "310p" in runners:
                return runners["310p"]
            return runners.get("default")
    return None


def _route_ut_dir(dir_path: str) -> PartitionKey:
    result = _resolve_partition(dir_path)
    return result if result is not None else _DEFAULT_KEY


def _route_e2e_dir(dir_path: str) -> PartitionKey | None:
    return _resolve_partition(dir_path)


def _route_e2e_file(file_path: str) -> PartitionKey | None:
    return _resolve_partition(file_path)


def _as_posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _pytest_node_file_path(path: str) -> str:
    """Return the real file path for a pytest nodeid target."""
    return path.split("::", 1)[0]


def _load_runners() -> dict[str, RunnerInfo]:
    with open(_RUNNER_LABEL_PATH) as f:
        raw = json.load(f)
    return {
        label: RunnerInfo(
            num_npus=info["npu_num"],
            npu_type=NpuType(info["chip"]),
            label=label,
            image_tag=info.get("image_tag", ""),
            csrc_cache_target=info.get("csrc_cache_target", ""),
        )
        for label, info in raw.items()
    }


def _get_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().splitlines() if f]


def _matches_path_dependency(file_path: str, dependency: str) -> bool:
    dep = dependency.rstrip("/")
    return file_path == dep or file_path.startswith(dep + "/")


def _as_base_list(base: str | list[str] | None) -> list[str]:
    if base is None:
        return []
    if isinstance(base, str):
        return [base]
    return base


def _merge_unique(parent: list[str], child: list[str]) -> list[str]:
    result = list(parent)
    for item in child:
        if item not in result:
            result.append(item)
    return result


def _resolve_config_inheritance(config: list[dict]) -> list[dict]:
    module_map = {m["name"]: m for m in config}
    resolved: dict[str, dict] = {}
    resolving: set[str] = set()
    inherited_fields = (
        "source_file_dependencies",
        "exclude_source_file_dependencies",
        "tests",
        "skip_tests",
    )

    def resolve(name: str) -> dict:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Circular test config inheritance detected for module: {name}")
        if name not in module_map:
            raise ValueError(f"Unknown base module in test config: {name}")

        resolving.add(name)
        module = dict(module_map[name])
        inherited_values = {field: [] for field in inherited_fields}
        for base_name in _as_base_list(module.get("base")):
            base_module = resolve(base_name)
            for field in inherited_fields:
                inherited_values[field] = _merge_unique(inherited_values[field], base_module.get(field, []))
        for field in inherited_fields:
            module[field] = _merge_unique(inherited_values[field], module.get(field, []))
        resolving.remove(name)
        resolved[name] = module
        return module

    return [resolve(module["name"]) for module in config]


def _match_modules(
    changed_files: list[str],
    config: list[dict],
    *,
    include_always: bool = True,
) -> list[str]:
    if not changed_files:
        return []
    matched: list[str] = []
    for module in config:
        if include_always and not module.get("optional", True):
            matched.append(module["name"])
            continue
        deps = module.get("source_file_dependencies", [])
        exclude_deps = module.get("exclude_source_file_dependencies", [])
        if any(
            _matches_path_dependency(f, dep)
            and not any(_matches_path_dependency(f, exclude) for exclude in exclude_deps)
            for f in changed_files
            for dep in deps
        ):
            matched.append(module["name"])
    return matched


def _collect_test_dirs(
    module_names: list[str],
    config: list[dict],
) -> tuple[list[str], list[str]]:
    """Collect test paths (directories or files) for the given modules.

    Returns (normal_dirs, cpu_only_dirs). *cpu_only_dirs* are from modules
    with ``cpu_only: true`` and should skip NPU convention subdirectories.

    Deduplicates parent/child paths: if both ``a/b`` and ``a/b/c`` are
    present, only ``a/b`` is kept.
    """
    module_map = {m["name"]: m for m in config}
    normal: set[str] = set()
    cpu_only: set[str] = set()
    for name in module_names:
        mod = module_map[name]
        target = cpu_only if mod.get("cpu_only") else normal
        for path in mod.get("tests", []):
            target.add(path.rstrip("/"))
    normal_list = _dedup_paths(normal)
    cpu_only_list = _dedup_paths(cpu_only)
    # Remove cpu_only paths that are already covered by a normal parent path
    cpu_only_list = [p for p in cpu_only_list if not any(p.startswith(n + "/") for n in normal_list)]
    return normal_list, cpu_only_list


def _dedup_paths(paths: set[str]) -> list[str]:
    sorted_paths = sorted(paths)
    result: list[str] = []
    for path in sorted_paths:
        if not any(path != other and path.startswith(other + "/") for other in sorted_paths):
            result.append(path)
    return result


def _configured_nodeid_targets_for_file(file_path: str, config: list[dict]) -> list[str]:
    file_path = _as_posix_path(_pytest_node_file_path(file_path))
    targets: list[str] = []
    seen: set[str] = set()
    for module in config:
        for test_target in module.get("tests", []):
            test_target = test_target.rstrip("/")
            if "::" not in test_target:
                continue
            target_file = _as_posix_path(_pytest_node_file_path(test_target))
            if target_file == file_path and test_target not in seen:
                targets.append(test_target)
                seen.add(test_target)
    return targets


def _is_skipped_test_target(target: str, skip_tests: set[str]) -> bool:
    target = target.rstrip("/")
    return target in skip_tests or _pytest_node_file_path(target) in skip_tests


def _is_ut_path(path: str) -> bool:
    return path == "tests/ut" or path.startswith("tests/ut/")


def _is_e2e_path(path: str) -> bool:
    return path == "tests/e2e" or path.startswith("tests/e2e/")


def _is_test_path(path: str) -> bool:
    return _is_ut_path(path) or _is_e2e_path(path)


def _is_cpu_ut_scoped_change(changed_files: list[str]) -> bool:
    """Return True when PR-filtered changes only require CPU UT.

    The workflow removes files outside its source-path filter before passing
    this list. Ordinary tools and test-only changes share the same broad CPU
    UT coverage; bisect paths retain their dedicated selection policy.
    """
    return bool(changed_files) and all(
        _is_test_path(f) or (_matches_path_dependency(f, "tools") and not _is_bisect_tool_scoped_path(f))
        for f in changed_files
    )


def _is_bisect_tool_scoped_path(file_path: str) -> bool:
    return file_path in _BISECT_TOOL_SUPPORT_FILES or any(
        _matches_path_dependency(file_path, root) for root in _BISECT_TOOL_ROOTS
    )


def _is_bisect_tool_scoped_change(changed_files: list[str]) -> bool:
    return (
        bool(changed_files)
        and any(_matches_path_dependency(f, root) for f in changed_files for root in _BISECT_TOOL_ROOTS)
        and all(_is_bisect_tool_scoped_path(f) for f in changed_files)
    )


def _scan_ut_test_dir(
    dir_path: str,
    groups: dict[PartitionKey, list[str]],
    cpu_only: bool = False,
) -> None:
    """Scan a UT directory and route tests by directory convention.

    Walks the directory tree. Each test file is routed individually based on
    its path — files under convention directories (e.g. ``a2/``, ``a3_2/``)
    go to the corresponding NPU runner, others go to the CPU group.

    If *cpu_only* is True, files under NPU convention directories are skipped.

    Always emits individual file paths to avoid test pollution when pytest
    runs a whole directory.
    """
    path = Path(_pytest_node_file_path(dir_path))
    if not path.exists():
        groups[_DEFAULT_KEY].append(dir_path)
        return

    if path.is_file():
        key = _route_ut_dir(dir_path)
        if cpu_only and key != _DEFAULT_KEY:
            print(
                f"Warning: cpu_only module test {dir_path} routes to NPU runner;"
                " check test_config.yaml for misconfigured cpu_only tests.",
                file=sys.stderr,
            )
            return
        groups[key].append(dir_path)
        return

    for f in sorted(path.rglob("test_*.py")):
        if "__pycache__" in f.parts:
            continue
        key = _route_ut_dir(str(f))
        if cpu_only and key != _DEFAULT_KEY:
            continue
        groups[key].append(str(f))


def _scan_e2e_test_dir(
    dir_path: str,
    groups: dict[PartitionKey, list[str]],
) -> None:
    """Scan an E2E directory or single file and route by directory convention.

    *dir_path* may be either a directory (all ``test_*.py`` under it are
    collected) or a single test file.
    """
    path = Path(_pytest_node_file_path(dir_path))
    if not path.exists():
        print(
            f"Warning: Path does not exist: {dir_path}",
            file=sys.stderr,
        )
        return

    if path.is_file():
        key = _route_e2e_file(dir_path)
        if key is not None:
            groups[key].append(dir_path)
        else:
            print(
                f"Warning: E2E test file {dir_path} does not match any runner pattern, skipping.",
                file=sys.stderr,
            )
        return

    key = _route_e2e_dir(dir_path + "/")
    if key is not None:
        test_files = sorted(str(f) for f in path.rglob("test_*.py"))
        if test_files:
            for f in test_files:
                f_key = _route_e2e_file(f)
                if f_key is not None:
                    groups[f_key].append(f)
        return

    for entry in sorted(path.iterdir()):
        if entry.is_dir():
            sub_key = _route_e2e_dir(str(entry) + "/")
            if sub_key is not None:
                test_files = sorted(str(f) for f in entry.rglob("test_*.py"))
                if test_files:
                    for f in test_files:
                        f_key = _route_e2e_file(f)
                        if f_key is not None:
                            groups[f_key].append(f)
            else:
                _scan_e2e_test_dir(str(entry), groups)


def _load_test_list_file(path: Path) -> list[str]:
    """Load pytest targets from *path*, one per non-empty, non-comment line."""
    targets: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            targets.append(_as_posix_path(line))
    return targets


def _route_explicit_test_target(
    target: str,
    groups: dict[PartitionKey, list[str]],
) -> None:
    """Route a single explicit UT/E2E target to the appropriate runner group."""
    # The coverage recommender emits "cpu-ut" (the batch label of the
    # default_cpu_ut module) instead of the real pytest path "tests/ut".
    # Map it back so the recommended path runs the same CPU UTs the
    # diff-based select-tests path selects for default_cpu_ut: tests/ut
    # scanned with cpu_only=True, i.e. NPU-convention subdirs are skipped
    # and the remaining files route to the CPU runner.
    cpu_only = False
    if target == CPU_UT_BATCH_ALIAS:
        target = CPU_UT_BATCH_PATH
        file_path = target
        cpu_only = True
    else:
        file_path = _pytest_node_file_path(target)
    if not _is_test_path(file_path):
        print(
            f"Warning: Skipping non-test path: {target}",
            file=sys.stderr,
        )
        return

    path = Path(file_path)
    if not path.exists():
        print(
            f"Warning: Path does not exist: {target}",
            file=sys.stderr,
        )
        return

    if _is_ut_path(file_path):
        if "::" in target or path.is_file():
            key = _route_ut_dir(file_path)
            if cpu_only and key != _DEFAULT_KEY:
                print(
                    f"Warning: cpu_only module test {target} routes to NPU runner;"
                    " check test_config.yaml for misconfigured cpu_only tests.",
                    file=sys.stderr,
                )
                return
            groups[key].append(target)
        else:
            _scan_ut_test_dir(target, groups, cpu_only=cpu_only)
        return

    if _is_e2e_path(file_path):
        _scan_e2e_test_dir(target, groups)
        return

    print(
        f"Warning: Skipping unrecognized test path: {target}",
        file=sys.stderr,
    )


def _dedup_groups(groups: dict[PartitionKey, list[str]]) -> None:
    """Deduplicate exact targets and file/nodeid containment across groups.

    A bare file target already executes every test in that file, so any
    ``file.py::nodeid`` target for the same file must be discarded. This is
    intentionally global: selection paths can add the two targets to different
    logical partitions before pinned routes are applied.
    """
    bare_files = {_as_posix_path(target) for tests in groups.values() for target in tests if "::" not in target}
    seen: set[str] = set()
    for key in groups:
        deduped: list[str] = []
        for target in groups[key]:
            normalized = _as_posix_path(target)
            if "::" in normalized and _pytest_node_file_path(normalized) in bare_files:
                continue
            if normalized in seen:
                continue
            deduped.append(target)
            seen.add(normalized)
        groups[key] = deduped


def _load_estimated_times(meta: dict) -> dict[str, float]:
    """Load per-test estimated times from the config meta dict.

    Tests not listed default to 600s when used by _partition_tests.
    """
    return {k: float(v) for k, v in meta.get("estimated_times", {}).items()}


def _load_partition_config(meta: dict) -> dict[PartitionKey, PartitionInfo]:
    """Load partition configuration from the config meta dict.

    Each logical partition key selects an exact label from runner_label.json
    and declares how many load-balanced groups to create.
    """
    result: dict[PartitionKey, PartitionInfo] = {}
    for key, value in (meta.get("partition", {}) or {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"Partition {key!r} must be a mapping with runner_label and count")
        runner_label = value.get("runner_label")
        if not isinstance(runner_label, str) or not runner_label:
            raise ValueError(f"Partition {key!r} must define a non-empty runner_label")
        count = int(value.get("count", 1))
        if count < 1:
            raise ValueError(f"Partition {key!r} count must be at least 1")
        result[str(key)] = PartitionInfo(
            runner_label=runner_label,
            count=count,
        )
    return result


def _load_pinned_routes(meta: dict) -> dict[str, PartitionKey]:
    """Load file-level routes applied after normal test selection.

    Each ``pinned_routes`` key is a destination logical partition. Tests stay
    classified by their existing technical directory and are moved only when
    they were actually selected by the normal module/explicit-test flow.
    """
    result: dict[str, PartitionKey] = {}
    for target_partition, value in (meta.get("pinned_routes", {}) or {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"Pinned route {target_partition!r} must be a mapping")
        tests = value.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"Pinned route {target_partition!r} must define a non-empty tests list")
        for test in tests:
            if not isinstance(test, str) or not test:
                raise ValueError(f"Pinned route {target_partition!r} contains an invalid test path")
            normalized = _as_posix_path(test.rstrip("/"))
            if "::" in normalized:
                raise ValueError(f"Pinned route {target_partition!r} must use file-level paths, not nodeids: {test}")
            if normalized in result:
                raise ValueError(f"Test path is configured in more than one pinned route: {normalized}")
            result[normalized] = str(target_partition)
    return result


def _validate_runner_config(
    runners: dict[str, RunnerInfo],
    partition_config: dict[PartitionKey, PartitionInfo],
    pinned_routes: dict[str, PartitionKey],
) -> None:
    unknown_labels = sorted(
        {info.runner_label for info in partition_config.values() if info.runner_label not in runners}
    )
    if unknown_labels:
        raise ValueError("Partition configuration references unknown runner label(s): " + ", ".join(unknown_labels))

    referenced_partitions = {partition_key for _, variants in _RUNNER_MAPPING for partition_key in variants.values()}
    referenced_partitions.update(pinned_routes.values())
    unknown_partitions = sorted(referenced_partitions - partition_config.keys())
    if unknown_partitions:
        raise ValueError(
            "Routing configuration references undefined logical partition(s): " + ", ".join(unknown_partitions)
        )


def _apply_pinned_routes(
    all_groups: dict[PartitionKey, list[str]],
    pinned_routes: dict[str, PartitionKey],
) -> dict[PartitionKey, list[str]]:
    """Move selected files (including selected nodeids) to pinned partitions."""
    if not pinned_routes:
        return all_groups

    result: dict[PartitionKey, list[str]] = defaultdict(list)
    for partition_key, tests in all_groups.items():
        for target in tests:
            file_path = _as_posix_path(_pytest_node_file_path(target))
            target_partition = pinned_routes.get(file_path)
            if target_partition is None:
                result[partition_key].append(target)
                continue
            result[target_partition].append(target)

    return result


def _apply_runner_label_override(
    all_groups: dict[PartitionKey, list[str]],
    runner_label: str,
    runners: dict[str, RunnerInfo],
    partition_config: dict[PartitionKey, PartitionInfo],
) -> tuple[dict[PartitionKey, list[str]], dict[PartitionKey, PartitionInfo]]:
    """Route every non-CPU group to an exact runner label.

    The override is intentionally transient: it does not need a static entry
    in ``test_config.yaml``. CPU groups retain their configured partitions.
    """
    override_runner = runners.get(runner_label)
    if override_runner is None:
        raise ValueError(f"Unknown runner label for --runner-label-override: {runner_label}")
    if override_runner.npu_type == NpuType.CPU:
        raise ValueError("--runner-label-override requires a non-CPU runner label")

    override_key = f"{override_runner.npu_type.value}-{override_runner.num_npus}"
    overridden: dict[PartitionKey, list[str]] = {}
    for partition_key, tests in all_groups.items():
        partition_info = partition_config.get(partition_key)
        runner = runners.get(partition_info.runner_label) if partition_info else None
        if runner is not None and runner.npu_type == NpuType.CPU:
            overridden[partition_key] = tests
        else:
            overridden.setdefault(override_key, []).extend(tests)

    updated_partition_config = dict(partition_config)
    updated_partition_config[override_key] = PartitionInfo(
        runner_label=runner_label,
        count=1,
    )
    return overridden, updated_partition_config


def _lookup_estimated_time(
    test_name: str,
    estimated_times: dict[str, float],
    default: float = 600.0,
) -> float:
    """Look up the estimated time for *test_name*, falling back to defaults.

    1. Try exact match (handles both file-level and ``::nodeid`` keys).
    2. Strip any ``::nodeid`` suffix and try again.
    3. Otherwise use *default*.

    File/nodeid containment is resolved by :func:`_dedup_groups` before this
    lookup is used. A nodeid selected on its own retains its exact estimate.
    """
    val = estimated_times.get(test_name)
    if val is not None:
        return val
    base = _pytest_node_file_path(test_name)
    if base != test_name:
        val = estimated_times.get(base)
        if val is not None:
            return val
    return default


def _partition_tests(
    tests: list[str],
    partition_size: int,
    estimated_times: dict[str, float],
) -> list[list[str]]:
    """Split *tests* into *partition_size* groups of roughly equal total time.

    Uses a greedy algorithm: sort tests descending by estimated time, then
    place each test into the currently lightest bucket.
    """
    if not tests or partition_size <= 1:
        return [tests]

    indexed = sorted(
        enumerate(tests),
        key=lambda x: (-_lookup_estimated_time(x[1], estimated_times), x[0]),
    )

    buckets: list[list[int]] = [[] for _ in range(partition_size)]
    sums = [0.0] * partition_size

    for idx, test in indexed:
        lightest = sums.index(min(sums))
        buckets[lightest].append(idx)
        sums[lightest] += _lookup_estimated_time(test, estimated_times)

    result = []
    for bucket in buckets:
        result.append(
            sorted(
                (tests[i] for i in bucket),
                key=lambda t: -_lookup_estimated_time(t, estimated_times),
            )
        )
    return result


def _build_test_group(
    runner: RunnerInfo,
    tests: list[str],
    partition_name: str,
    partition: str,
) -> dict:
    group: dict = {
        "num_npus": runner.num_npus,
        "npu_type": runner.npu_type.value,
        "runner": runner.label,
        "tests": " ".join(sorted(tests)),
        "partition_name": partition_name,
        "partition": partition,
    }
    if runner.image_tag:
        group["image_tag"] = runner.image_tag
    if runner.csrc_cache_target:
        group["csrc_cache_target"] = runner.csrc_cache_target
    return group


def _resolve_to_runners(
    all_groups: dict[PartitionKey, list[str]],
    runners: dict[str, RunnerInfo],
    partition_config: dict[PartitionKey, PartitionInfo],
    estimated_times: dict[str, float] | None = None,
) -> list[dict]:
    result: list[dict] = []
    estimated_times = estimated_times or {}

    def partition_sort_key(item: tuple[PartitionKey, list[str]]) -> tuple[int, str, str]:
        partition_key = item[0]
        partition_info = partition_config[partition_key]
        runner = runners[partition_info.runner_label]
        return (runner.num_npus, runner.npu_type.value, partition_key)

    nonempty_groups = (item for item in all_groups.items() if item[1])
    for partition_key, tests in sorted(nonempty_groups, key=partition_sort_key):
        partition_info = partition_config[partition_key]
        runner = runners[partition_info.runner_label]

        psize = partition_info.count

        if psize > 1:
            buckets = _partition_tests(sorted(tests), psize, estimated_times)
            for i, bucket in enumerate(buckets):
                if not bucket:
                    continue
                result.append(
                    _build_test_group(
                        runner,
                        bucket,
                        partition_key,
                        f"{i + 1}-{psize}",
                    )
                )
        else:
            result.append(_build_test_group(runner, tests, partition_key, "1-1"))

    return result


def _write_output(
    test_groups: list[dict],
    matched_modules: list[str],
) -> None:
    has_tests = len(test_groups) > 0
    groups_json = json.dumps(test_groups, separators=(",", ":"))
    cache_target_ids = sorted({group["csrc_cache_target"] for group in test_groups if group.get("csrc_cache_target")})

    outputs = {
        "test_groups": groups_json,
        "has_tests": str(has_tests).lower(),
        "csrc_cache_target_ids": json.dumps(cache_target_ids, separators=(",", ":")),
        "matched_modules": ",".join(matched_modules),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")

    _print_summary(test_groups, matched_modules, has_tests)


def _print_summary(
    test_groups: list[dict],
    matched_modules: list[str],
    has_tests: bool,
) -> None:
    divider = "=" * 60
    print(f"\n{divider}", file=sys.stderr)
    print("Selective Test Scope Summary", file=sys.stderr)
    print(divider, file=sys.stderr)
    print(f"Matched modules: {matched_modules or '(none)'}", file=sys.stderr)
    print(f"Has tests to run: {has_tests}", file=sys.stderr)

    for group in test_groups:
        runner = group["runner"]
        tests = group["tests"].split()
        partition_name = group["partition_name"]
        partition_info = group.get("partition", "full")
        header = f"### {partition_name} ({len(tests)} tests) part {partition_info} -> `{runner}`"
        print(f"\n  {header}", file=sys.stderr)
        for t in tests:
            print(f"    - {t}", file=sys.stderr)

    print(f"{divider}\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Determine test scope from changed files or explicit e2e test paths",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--changed-files",
        nargs="+",
        help="List of changed file paths",
    )
    input_group.add_argument(
        "--diff-base",
        type=str,
        help="Git ref to diff against (e.g. origin/main)",
    )
    input_group.add_argument(
        "--explicit-e2e-tests",
        nargs="+",
        help="List of explicit e2e test paths (files or directories) to run. "
        "Bypasses module matching and routes each path to the appropriate runner. "
        "Use this for the /e2e slash command to run a specific subset of tests. "
        "Supports ``::nodeid`` suffix (e.g. ``test_foo.py::TestClass::test_method``) "
        "to run a single test method.",
    )
    input_group.add_argument(
        "--test-list-file",
        type=Path,
        help="Path to a text file listing pytest targets to run (one per line). "
        "Supports UT and E2E paths, directories, and ``::nodeid`` suffixes for "
        "test classes or methods. Blank lines and ``#`` comments are ignored.",
    )
    parser.add_argument(
        "--filtered-changed-files-json",
        type=str,
        default=None,
        help="JSON array containing changed files matched by the PR workflow path filter. "
        "Used only for test-scoped detection and changed-test collection.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to test_config.yaml",
    )
    parser.add_argument(
        "--run-all-modules",
        action="store_true",
        help="Run tests for all configured modules regardless of changed files",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Force-select specific modules (comma-separated), bypassing file-change matching",
    )
    parser.add_argument(
        "--runner-label-override",
        type=str,
        default=None,
        help="Force route all non-CPU tests to an exact label from runner_label.json",
    )
    args = parser.parse_args()
    docs = list(yaml.safe_load_all(args.config.read_text()))
    config = _resolve_config_inheritance(docs[0])
    meta = docs[1] if len(docs) >= 2 and docs[1] else {}
    runners = _load_runners()
    partition_config = _load_partition_config(meta)
    estimated_times = _load_estimated_times(meta)
    _load_runner_mapping(meta)
    pinned_routes = _load_pinned_routes(meta)
    _validate_runner_config(runners, partition_config, pinned_routes)

    skip_tests: set[str] = set()
    for module in config:
        for s in module.get("skip_tests", []):
            skip_tests.add(s.rstrip("/"))

    if args.explicit_e2e_tests:
        matched_modules: list[str] = []
        all_groups: dict[PartitionKey, list[str]] = defaultdict(list)
        for path in args.explicit_e2e_tests:
            if not _is_e2e_path(_pytest_node_file_path(path)):
                print(
                    f"Warning: Skipping non-e2e path: {path}",
                    file=sys.stderr,
                )
                continue
            _scan_e2e_test_dir(path, all_groups)
    elif args.test_list_file:
        if not args.test_list_file.is_file():
            print(
                f"ERROR: Test list file does not exist: {args.test_list_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        matched_modules = []
        all_groups = defaultdict(list)
        explicit_targets = _load_test_list_file(args.test_list_file)
        if not explicit_targets:
            print(
                f"Warning: Test list file is empty: {args.test_list_file}",
                file=sys.stderr,
            )
        for target in explicit_targets:
            _route_explicit_test_target(target, all_groups)
        _dedup_groups(all_groups)
    else:
        changed_files = _get_changed_files(args.diff_base) if args.diff_base else args.changed_files
        filtered_changed_files = changed_files
        if args.filtered_changed_files_json is not None:
            try:
                filtered_changed_files = json.loads(args.filtered_changed_files_json)
            except json.JSONDecodeError as exc:
                parser.error(f"Invalid --filtered-changed-files-json: {exc}")
            if not isinstance(filtered_changed_files, list) or not all(
                isinstance(file_path, str) for file_path in filtered_changed_files
            ):
                parser.error("--filtered-changed-files-json must be a JSON array of strings")
        bisect_tool_scoped_change = _is_bisect_tool_scoped_change(changed_files)
        cpu_ut_scoped_change = _is_cpu_ut_scoped_change(filtered_changed_files)
        if bisect_tool_scoped_change:
            print(
                "Detected bisect tool-scoped change: running only matching tool modules (skipping always-on modules).",
                file=sys.stderr,
            )
        elif cpu_ut_scoped_change:
            print(
                "Detected CPU-UT-scoped change: running only default_cpu_ut"
                " and the changed test files (skipping source-driven modules).",
                file=sys.stderr,
            )
        if args.run_all_modules:
            matched_modules = [module["name"] for module in config]
        elif args.modules:
            requested = set(args.modules.split(","))
            available = {m["name"] for m in config}
            missing = requested - available
            if missing:
                print(
                    f"ERROR: unknown module(s): {', '.join(sorted(missing))}. "
                    f"Available: {', '.join(sorted(available))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            matched_modules = sorted(requested)
        elif bisect_tool_scoped_change:
            matched_modules = _match_modules(changed_files, config, include_always=False)
        elif cpu_ut_scoped_change:
            matched_modules = [m["name"] for m in config if m["name"] == DEFAULT_CPU_UT_MODULE]
        else:
            matched_modules = _match_modules(changed_files, config)
        test_dirs, cpu_only_dirs = _collect_test_dirs(matched_modules, config)

        changed_test_files = []
        for f in filtered_changed_files:
            if not _is_test_path(f):
                continue
            target = Path(_pytest_node_file_path(f))
            if target.name.startswith("test_") and target.exists():
                changed_test_files.append(f)

        ut_dirs = [d for d in test_dirs if _is_ut_path(d)]
        cpu_only_ut_dirs = [d for d in cpu_only_dirs if _is_ut_path(d)]
        e2e_dirs = [d for d in test_dirs if _is_e2e_path(d)]

        all_groups: dict[PartitionKey, list[str]] = defaultdict(list)

        for dir_path in ut_dirs:
            p = Path(_pytest_node_file_path(dir_path))
            if p.is_file():
                key = _route_ut_dir(dir_path)
                all_groups[key].append(dir_path)
            else:
                _scan_ut_test_dir(dir_path, all_groups)
        for dir_path in cpu_only_ut_dirs:
            p = Path(_pytest_node_file_path(dir_path))
            if p.is_file():
                key = _route_ut_dir(dir_path)
                if key == _DEFAULT_KEY:
                    all_groups[key].append(dir_path)
            else:
                _scan_ut_test_dir(dir_path, all_groups, cpu_only=True)

        for dir_path in e2e_dirs:
            _scan_e2e_test_dir(dir_path, all_groups)

        for changed_test_file in changed_test_files:
            if "::" in changed_test_file:
                changed_targets = [changed_test_file]
            else:
                changed_targets = _configured_nodeid_targets_for_file(changed_test_file, config) or [changed_test_file]
            for f in changed_targets:
                if _is_skipped_test_target(f, skip_tests):
                    continue
                if _is_ut_path(f):
                    key = _route_ut_dir(f)
                    all_groups[key].append(f)
                elif _is_e2e_path(f):
                    key = _route_e2e_file(f)
                    if key is not None:
                        all_groups[key].append(f)

        _dedup_groups(all_groups)

    if skip_tests:
        for key in list(all_groups.keys()):
            filtered: list[str] = []
            for t in all_groups[key]:
                if _is_skipped_test_target(t, skip_tests):
                    continue
                p = Path(_pytest_node_file_path(t))
                if p.is_dir():
                    sub = [
                        str(f) for f in sorted(p.rglob("test_*.py")) if not _is_skipped_test_target(str(f), skip_tests)
                    ]
                    if sub:
                        filtered.extend(sub)
                else:
                    filtered.append(t)
            all_groups[key] = filtered
        _dedup_groups(all_groups)

    # Normalize every input mode before pinning. A bare file contains all of
    # its nodeids, even when the targets came from different module lists.
    _dedup_groups(all_groups)
    all_groups = _apply_pinned_routes(all_groups, pinned_routes)

    # A command-line override is an explicit, transient choice and therefore
    # has higher priority than static pinned routes.
    if args.runner_label_override:
        try:
            all_groups, partition_config = _apply_runner_label_override(
                all_groups,
                args.runner_label_override,
                runners,
                partition_config,
            )
        except ValueError as exc:
            parser.error(str(exc))

    test_groups = _resolve_to_runners(all_groups, runners, partition_config, estimated_times)

    _write_output(test_groups, matched_modules)


if __name__ == "__main__":
    main()
