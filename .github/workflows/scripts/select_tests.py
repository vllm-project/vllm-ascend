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

Pipeline:
  1. Diff       -- get changed files from git.
  2. Match      -- identify affected modules via test_config.yaml.
  3. Collect    -- gather test paths (always resolved to individual files).
  4. Route      -- determine runner via config-driven runner_mapping.
  5. Partition  -- split test groups across parallel runners by estimated time.
  6. Output     -- write test_groups / has_tests / matched_modules.

Routing is driven by ``test_config.yaml`` ``runner_mapping:`` (regex patterns).
Partition sizing by ``partition:`` config block.
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
    CPU = "cpu"


@dataclass
class RunnerInfo:
    num_npus: int
    npu_type: NpuType
    label: str
    image_tag: str = ""


RunnerKey = tuple[int, NpuType]
_DEFAULT_KEY: RunnerKey = (0, NpuType.CPU)

# Populated by _load_runner_mapping(). Ordered list of (regex, {key: RunnerKey}).
_RUNNER_MAPPING: list[tuple[re.Pattern, dict[str, RunnerKey]]] = []


def _parse_runner_key(runner_key: str) -> RunnerKey:
    """Parse ``a2_x1`` → ``(1, NpuType.A2)``, ``310p_x4`` → ``(4, NpuType._310P)``."""
    parts = runner_key.rsplit("_x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid runner key: {runner_key!r}")
    raw_type, raw_npus = parts
    npu_type = NpuType(raw_type)
    num_npus = int(raw_npus)
    return (num_npus, npu_type)


def _load_runner_mapping(config_path: Path) -> None:
    """Load runner mapping from the second YAML document into ``_RUNNER_MAPPING``.

    Config format::

        runner_mapping:
          <regex_pattern>:
            default: <runner_key>
            "310p": <runner_key>   # optional override for 310P files

    Patterns are sorted longest first so more specific patterns match first.
    """
    global _RUNNER_MAPPING
    _RUNNER_MAPPING = []
    try:
        docs = list(yaml.safe_load_all(config_path.read_text()))
        if len(docs) >= 2:
            meta = docs[1] or {}
            raw = list((meta.get("runner_mapping", {}) or {}).items())
            raw.sort(key=lambda x: -len(x[0]))
            for pattern_str, runner_config in raw:
                runners: dict[str, RunnerKey] = {}
                for key, val in runner_config.items():
                    runners[key] = _parse_runner_key(val)
                _RUNNER_MAPPING.append((re.compile(pattern_str), runners))
    except Exception:
        pass


def _resolve_runner(file_path: str) -> RunnerKey | None:
    """Match *file_path* against ``_RUNNER_MAPPING``.

    Returns the ``default`` runner for the first matching pattern.
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


def _route_ut_dir(dir_path: str) -> RunnerKey:
    result = _resolve_runner(dir_path)
    return result if result is not None else _DEFAULT_KEY


def _route_e2e_dir(dir_path: str) -> RunnerKey | None:
    return _resolve_runner(dir_path)


def _route_e2e_file(file_path: str) -> RunnerKey | None:
    return _resolve_runner(file_path)


def _as_posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _pytest_node_file_path(path: str) -> str:
    """Return the real file path for a pytest nodeid target."""
    return path.split("::", 1)[0]


def _load_runners() -> list[RunnerInfo]:
    with open(_RUNNER_LABEL_PATH) as f:
        raw = json.load(f)
    return [
        RunnerInfo(
            num_npus=info["npu_num"],
            npu_type=NpuType(info["chip"]),
            label=label,
            image_tag=info.get("image_tag", ""),
        )
        for label, info in raw.items()
    ]


def _get_changed_files(base_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def _matches_path_dependency(file_path: str, dependency: str) -> bool:
    dep = dependency.rstrip("/")
    return file_path == dep or file_path.startswith(dep + "/")


def _as_base_list(base: str | list[str] | None) -> list[str]:
    if base is None:
        return []
    if isinstance(base, str):
        return [base]
    return base


def _merge_unique(parent: list, child: list) -> list:
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
) -> list[str]:
    if not changed_files:
        return []
    matched: list[str] = []
    for module in config:
        if not module.get("optional", True):
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


def _scan_ut_test_dir(
    dir_path: str,
    groups: dict[RunnerKey, list[str]],
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
            return
        groups[key].append(dir_path)
        return

    for f in sorted(path.rglob("test_*.py")):
        if any(part in ("__pycache__",) for part in f.parts):
            continue
        key = _route_ut_dir(str(f))
        if cpu_only and key != _DEFAULT_KEY:
            continue
        groups[key].append(str(f))


def _scan_e2e_test_dir(
    dir_path: str,
    groups: dict[RunnerKey, list[str]],
) -> None:
    """Scan an E2E directory or single file and route by directory convention.

    *dir_path* may be either a directory (all ``test_*.py`` under it are
    collected) or a single test file.
    """
    path = Path(_pytest_node_file_path(dir_path))
    if not path.exists():
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


def _dedup_groups(groups: dict[RunnerKey, list[str]]) -> None:
    for key in groups:
        seen: set[str] = set()
        deduped: list[str] = []
        for target in groups[key]:
            if target not in seen:
                deduped.append(target)
                seen.add(target)
        groups[key] = deduped


def _find_runner(
    num_npus: int,
    npu_type: NpuType,
    runners: list[RunnerInfo],
) -> RunnerInfo | None:
    if npu_type == NpuType.CPU:
        candidates = [r for r in runners if r.npu_type == NpuType.CPU]
    else:
        candidates = [r for r in runners if r.npu_type == npu_type and r.num_npus == num_npus]
    return candidates[0] if candidates else None


def _load_estimated_times(
    config_path: Path,
) -> dict[str, float]:
    """Load per-test estimated times from the second YAML document.

    Tests not listed default to 600s when used by _partition_tests.
    """
    estimated_times: dict[str, float] = {}
    try:
        docs = list(yaml.safe_load_all(config_path.read_text()))
        if len(docs) >= 2:
            meta = docs[1] or {}
            for k, v in meta.get("estimated_times", {}).items():
                estimated_times[k] = float(v)
    except Exception:
        pass
    return estimated_times


def _load_partition_config(
    config_path: Path,
) -> dict[str, int]:
    """Load partition configuration from the second YAML document.

    Returns a dict mapping runner keys (e.g. ``a2_x1``) to partition
    counts.  Runner keys not listed default to 1.
    """
    partition: dict[str, int] = {}
    try:
        docs = list(yaml.safe_load_all(config_path.read_text()))
        if len(docs) >= 2:
            meta = docs[1] or {}
            partition = {k: int(v) for k, v in meta.get("partition", {}).items()}
    except Exception:
        pass
    return partition


def _lookup_estimated_time(
    test_name: str,
    estimated_times: dict[str, float],
    default: float = 600.0,
) -> float:
    """Look up the estimated time for *test_name*, falling back to defaults.

    1. Try exact match (handles both file-level and ``::nodeid`` keys).
    2. Strip any ``::nodeid`` suffix and try again.
    3. Otherwise use *default*.

    Note: when both a file-level path and a ``::nodeid`` path for the same
    file exist in module ``tests:`` lists, that method executes twice.
    Avoid mixing levels for the same file in ``tests:``.
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


def _resolve_to_runners(
    all_groups: dict[RunnerKey, list[str]],
    runners: list[RunnerInfo],
    partition_config: dict[str, int] | None = None,
    estimated_times: dict[str, float] | None = None,
) -> list[dict]:
    result: list[dict] = []
    errors: list[str] = []
    partition_config = partition_config or {}
    estimated_times = estimated_times or {}

    for (num_npus, npu_type), tests in sorted(all_groups.items()):
        if not tests:
            continue
        runner = _find_runner(num_npus, npu_type, runners)
        if runner is None:
            available = [f"{r.label} ({r.npu_type.value} x{r.num_npus})" for r in runners if r.npu_type == npu_type]
            header = f"\n  Runner key ({npu_type.value} x{num_npus}) -- no runner available."
            runners_line = (
                f"\n    Available {npu_type.value} runners: {', '.join(available)}"
                if available
                else f'\n    No runners defined for chip type "{npu_type.value}".'
            )
            tests_line = "\n    Affected tests:\n" + "\n".join(f"      - {t}" for t in sorted(tests))
            errors.append(header + runners_line + tests_line)
            continue

        partition_key = f"{npu_type.value}_x{num_npus}"
        psize = partition_config.get(partition_key, 1)

        if psize > 1:
            buckets = _partition_tests(sorted(tests), psize, estimated_times)
            for i, bucket in enumerate(buckets):
                if not bucket:
                    continue
                group: dict = {
                    "num_npus": num_npus,
                    "npu_type": npu_type.value,
                    "runner": runner.label,
                    "tests": " ".join(sorted(bucket)),
                    "partition": f"{i + 1}-{psize}",
                }
                if runner.image_tag:
                    group["image_tag"] = runner.image_tag
                result.append(group)
        else:
            group = {
                "num_npus": num_npus,
                "npu_type": npu_type.value,
                "runner": runner.label,
                "tests": " ".join(sorted(tests)),
                "partition": "1-1",
            }
            if runner.image_tag:
                group["image_tag"] = runner.image_tag
            result.append(group)

    if errors:
        print(
            "\nERROR: The following test groups cannot be routed to any runner"
            " in runner_label.json:" + "".join(errors) + "\n\nPlease fix the directory structure or add the"
            " missing runner to runner_label.json.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return result


def _write_output(
    test_groups: list[dict],
    matched_modules: list[str],
) -> None:
    has_tests = len(test_groups) > 0
    groups_json = json.dumps(test_groups, separators=(",", ":"))

    outputs = {
        "test_groups": groups_json,
        "has_tests": str(has_tests).lower(),
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
        npu_type = group["npu_type"]
        num_npus = group["num_npus"]
        runner = group["runner"]
        tests = group["tests"].split()
        partition_info = group.get("partition", "full")
        if npu_type == "cpu":
            header = f"### CPU ({len(tests)} tests) part {partition_info} -> `{runner}`"
        else:
            header = f"### {npu_type.upper()} x{num_npus} ({len(tests)} tests) part {partition_info} -> `{runner}`"
        print(f"\n  {header}", file=sys.stderr)
        for t in tests:
            print(f"    - {t}", file=sys.stderr)

    print(f"{divider}\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Determine test scope based on changed files",
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

    args = parser.parse_args()
    _load_runner_mapping(args.config)
    config = _resolve_config_inheritance(next(yaml.safe_load_all(args.config.read_text())))

    changed_files = _get_changed_files(args.diff_base) if args.diff_base else args.changed_files
    matched_modules = (
        [module["name"] for module in config] if args.run_all_modules else _match_modules(changed_files, config)
    )
    test_dirs, cpu_only_dirs = _collect_test_dirs(matched_modules, config)

    skip_tests: set[str] = set()
    for module in config:
        for s in module.get("skip_tests", []):
            skip_tests.add(s.rstrip("/"))

    changed_test_files = [
        f
        for f in changed_files
        if (_is_ut_path(f) or _is_e2e_path(f))
        and Path(_pytest_node_file_path(f)).name.startswith("test_")
        and Path(_pytest_node_file_path(f)).exists()
    ]

    ut_dirs = [d for d in test_dirs if _is_ut_path(d)]
    cpu_only_ut_dirs = [d for d in cpu_only_dirs if _is_ut_path(d)]
    e2e_dirs = [d for d in test_dirs if _is_e2e_path(d)]

    all_groups: dict[RunnerKey, list[str]] = defaultdict(list)

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

    runners = _load_runners()
    estimated_times = _load_estimated_times(args.config)
    partition_config = _load_partition_config(args.config)
    test_groups = _resolve_to_runners(all_groups, runners, partition_config, estimated_times)

    _write_output(test_groups, matched_modules)


if __name__ == "__main__":
    main()
