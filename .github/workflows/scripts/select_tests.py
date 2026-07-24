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

Pipeline (PR-driven mode):
  1. Diff       -- get changed files from git.
  2. Match      -- identify affected modules via test_config.yaml.
  3. Collect    -- gather test paths (always resolved to individual files).
  4. Route      -- determine runner via config-driven runner_mapping.
  5. Partition  -- split test groups across parallel runners by estimated time.
  6. Output     -- write test_groups / has_tests / matched_modules.

Test-only optimization:
  If a PR changes only files under ``tests/`` (no source code touched), the
  module-matching step is bypassed. Only the ``default_cpu_ut`` module (which
  is always-on) and the changed test files themselves are run. This avoids
  the broad regression triggered by ``optional: false`` modules when the
  intent of the PR is purely to add or adjust tests.

Bisect-tool optimization:
  If a PR is scoped to ``tools/bisect`` and its paired UT/config/format files,
  the always-on modules are skipped and only modules whose dependencies match
  the changed files are selected. This keeps maintenance-only tool changes
  from triggering the full CPU and NPU regression suite.

Routing is driven by ``test_config.yaml`` ``runner_mapping:`` (regex patterns).
Partition sizing by ``partition:`` config block.
See ``test_config.yaml`` for details.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
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


@dataclass(frozen=True)
class RunnerInfo:
    num_npus: int
    npu_type: NpuType
    label: str
    image_tag: str = ""


RunnerKey = tuple[int, NpuType]
_DEFAULT_KEY: RunnerKey = (0, NpuType.CPU)

# The always-on CPU UT module. In test-only changes, only this module
# is selected for UT runs (along with the changed test files).
DEFAULT_CPU_UT_MODULE = "default_cpu_ut"

_BISECT_TOOL_ROOTS = ("tools/bisect", "tests/ut/tools/bisect")
_BISECT_TOOL_SUPPORT_FILES = {
    ".github/workflows/scripts/select_tests.py",
    ".github/workflows/scripts/test_config.yaml",
    ".github/workflows/scripts/test_select_tests.py",
    "csrc/build.sh",
    "tests/ut/tools/__init__.py",
}

# Populated by _load_runner_mapping(). Ordered list of (regex, {key: RunnerKey}).
_RUNNER_MAPPING: list[tuple[re.Pattern, dict[str, RunnerKey]]] = []

# Populated by _load_marker_routing(). E.g.:
#   {"requires_hardware": {"A2": "a2", "A3": "a3"},
#    "requires_npus": {"1": "x1", "2": "x2", "4": "x4"}}
_MARKER_ROUTING: dict[str, dict[str, str]] = {}

# Coarse hardware types that don't need a specific SOC runner.
# Any requires_hardware value outside this set is treated as SOC-specific.
_COARSE_HW_TYPES: frozenset = frozenset({"A2", "A3", "310P", "A5"})


def _is_soc_specific_test(file_path: str) -> bool:
    """Return True if the test file has a SOC-specific ``requires_hardware``
    marker (e.g. ``ascend910_9392``) rather than a coarse type (``A3``)."""
    markers = _parse_test_markers(file_path)
    if markers is None:
        return False
    hw_values = markers.get("requires_hardware")
    if not hw_values:
        return False
    return any(v not in _COARSE_HW_TYPES for v in hw_values)


def _parse_runner_key(runner_key: str) -> RunnerKey:
    """Parse ``a2_x1`` → ``(1, NpuType.A2)``, ``310p_x4`` → ``(4, NpuType._310P)``."""
    parts = runner_key.rsplit("_x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid runner key: {runner_key!r}")
    raw_type, raw_npus = parts
    npu_type = NpuType(raw_type)
    num_npus = int(raw_npus)
    return (num_npus, npu_type)


def _load_runner_mapping(meta: dict) -> None:
    """Load runner mapping from the config meta dict into ``_RUNNER_MAPPING``.

    Config format::

        runner_mapping:
          <regex_pattern>:
            default: <runner_key>
            "310p": <runner_key>   # optional override for 310P files

    Patterns are sorted longest first so more specific patterns match first.
    """
    global _RUNNER_MAPPING
    _RUNNER_MAPPING = []
    raw = list((meta.get("runner_mapping", {}) or {}).items())
    raw.sort(key=lambda x: -len(x[0]))
    for pattern_str, runner_config in raw:
        runners: dict[str, RunnerKey] = {}
        for key, val in runner_config.items():
            runners[key] = _parse_runner_key(val)
        _RUNNER_MAPPING.append((re.compile(pattern_str), runners))


def _load_marker_routing(meta: dict) -> None:
    """Load marker routing from the config meta dict into ``_MARKER_ROUTING``.

    The marker_routing config maps pytest marker argument values to
    runner-key components::

        marker_routing:
          requires_hardware:
            A2: a2
            A3: a3
          requires_npus:
            1: x1
            2: x2
            4: x4
    """
    global _MARKER_ROUTING
    _MARKER_ROUTING = {}
    raw = meta.get("marker_routing", {}) or {}
    for marker_name, arg_map in raw.items():
        _MARKER_ROUTING[marker_name] = {str(k): str(v) for k, v in (arg_map or {}).items()}


def _resolve_runner(file_path: str) -> RunnerKey | None:
    """Match *file_path* against ``_RUNNER_MAPPING``.

    Returns the ``default`` runner for the first matching pattern.
    Supports generic filename-based overrides: if the file name
    contains ``_<suffix>`` (e.g. ``_a2``, ``_a3``, ``_310p``) and
    the matched pattern has a matching entry, that entry is returned
    instead of the default.
    """
    route_path = _as_posix_path(_pytest_node_file_path(file_path))
    name = Path(route_path).name
    for pattern, runners in _RUNNER_MAPPING:
        if pattern.search(route_path):
            # Try filename-based suffix override (e.g. _a2, _a3, _310p)
            for suffix, runner_key in runners.items():
                if suffix == "default":
                    continue
                if f"_{suffix}" in name:
                    return runner_key
            return runners.get("default")
    return None


def _route_ut_dir(dir_path: str) -> RunnerKey:
    result = _resolve_runner(dir_path)
    return result if result is not None else _DEFAULT_KEY


def _route_e2e_dir(dir_path: str) -> RunnerKey | None:
    return _resolve_runner(dir_path)


def _route_e2e_file(file_path: str) -> RunnerKey | None:
    return _resolve_runner(file_path)


def _parse_test_markers(file_path: str) -> dict[str, list[str]] | None:
    """Parse ``@pytest.mark.requires_hardware`` and
    ``@pytest.mark.requires_npus`` from a test file using AST.

    Returns a dict like ``{"requires_hardware": ["A2", "A3"],
    "requires_npus": ["1"]}``, or ``None`` if the file has no such
    markers or cannot be parsed.

    Only string literal arguments are extracted (no variable references).
    """
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError):
        return None

    markers: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            info = _extract_marker_arg(decorator, "requires_hardware", "requires_npus")
            if info:
                name, args = info
                markers.setdefault(name, []).extend(args)

    return markers if markers else None


def _extract_marker_arg(
    decorator: ast.AST,
    *marker_names: str,
) -> tuple[str, list[str]] | None:
    """If *decorator* is a ``@pytest.mark.<name>(...)`` call with
    *name* in *marker_names*, return ``(name, [str_args...])``.
    """
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in marker_names:
        return None
    if not isinstance(func.value, ast.Attribute):
        return None
    if func.value.attr != "mark":
        return None

    args: list[str] = []
    for arg in decorator.args:
        if isinstance(arg, ast.Constant):
            if isinstance(arg.value, str):
                args.append(arg.value)
            elif isinstance(arg.value, int):
                args.append(str(arg.value))
    return (func.attr, args)


def _resolve_runner_by_markers(
    file_path: str,
) -> list[RunnerKey] | None:
    """Try to route *file_path* by its ``@pytest.mark.requires_*``
    markers, using ``_MARKER_ROUTING`` config to map marker values to
    runner-key components.

    Returns a list of ``RunnerKey``\s if markers were found and
    successfully resolved, or ``None`` if no markers exist (caller
    should fall back to directory-based routing).
    """
    markers = _parse_test_markers(file_path)
    if markers is None:
        return None

    hw_values = markers.get("requires_hardware")
    npus_values = markers.get("requires_npus")

    if hw_values is None and npus_values is None:
        return None

    # Map marker values to runner-key components
    hw_routing = _MARKER_ROUTING.get("requires_hardware", {})
    npus_routing = _MARKER_ROUTING.get("requires_npus", {})

    chips: set[NpuType] = set()
    npu_counts: set[int] = set()

    if hw_values:
        for hw in hw_values:
            chip_key = hw_routing.get(hw)
            if chip_key:
                with contextlib.suppress(ValueError):
                    chips.add(NpuType(chip_key))

    if npus_values:
        for n in npus_values:
            card_key = npus_routing.get(str(n))
            if card_key:
                # card_key is e.g. "x1", "x2", "x4"
                with contextlib.suppress(ValueError, IndexError):
                    npu_counts.add(int(card_key[1:]))

    # Only use marker-based routing when *both* chip and NPU count
    # are explicitly specified.  An incomplete annotation (e.g. only
    # requires_hardware without requires_npus) falls back to the
    # original directory-based routing for backward compatibility.
    if not chips or not npu_counts:
        return None

    result: list[RunnerKey] = []
    for chip in chips:
        for n in sorted(npu_counts):
            result.append((n, chip))

    return result


def _test_routes_to_runner(file_path: str, target: RunnerKey) -> bool:
    """Return True if *file_path* would be routed to *target* runner.

    Priority:
    1. If the file has complete markers (both ``requires_hardware`` and
       ``requires_npus``), check compatibility with *target*.
    2. Otherwise fall back to directory-based routing (``_route_e2e_file``).
    """
    marker_keys = _resolve_runner_by_markers(file_path)
    if marker_keys is not None:
        return target in marker_keys
    return _route_e2e_file(file_path) == target


def _discover_tests_for_runner(
    test_paths: list[str],
    target: RunnerKey,
) -> list[str]:
    """Discover test files in *test_paths* that route to *target* runner.

    Each path may be a single file (``test_foo.py``) or a directory
    (all ``test_*.py`` files under it are examined).  Returns a sorted
    list of paths that would be routed to the given runner.
    """
    result: list[str] = []
    for entry in test_paths:
        path = Path(_pytest_node_file_path(entry))
        if not path.exists():
            continue
        if path.is_file():
            if _test_routes_to_runner(str(path), target):
                result.append(str(path))
        else:
            for f in sorted(path.rglob("test_*.py")):
                if "__pycache__" in f.parts:
                    continue
                if _test_routes_to_runner(str(f), target):
                    result.append(str(f))
    return result


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


def _is_test_only_change(changed_files: list[str]) -> bool:
    """Return True if *changed_files* contains only files under ``tests/``.

    When a PR touches nothing but test files, there is no source change
    requiring broad regression; only the changed tests (and the always-on
    ``default_cpu_ut`` module) need to run.
    """
    return bool(changed_files) and all(_is_test_path(f) for f in changed_files)


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
    groups: dict[RunnerKey, list[str]],
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
        # Try marker-based routing first
        marker_keys = _resolve_runner_by_markers(dir_path)
        if marker_keys is not None:
            for key in marker_keys:
                groups[key].append(dir_path)
            return
        # Fall back to directory-based routing
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
                # Try marker-based routing per file
                marker_keys = _resolve_runner_by_markers(f)
                if marker_keys is not None:
                    for mk in marker_keys:
                        groups[mk].append(f)
                else:
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
                        # Try marker-based routing per file
                        marker_keys = _resolve_runner_by_markers(f)
                        if marker_keys is not None:
                            for mk in marker_keys:
                                groups[mk].append(f)
                        else:
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
    label_suffix: str = "",
) -> RunnerInfo | None:
    if npu_type == NpuType.CPU:
        candidates = [r for r in runners if r.npu_type == NpuType.CPU]
    else:
        candidates = [r for r in runners if r.npu_type == npu_type and r.num_npus == num_npus]
    if label_suffix:
        # Prefer runners whose label ends with the given suffix
        preferred = [r for r in candidates if r.label.endswith(label_suffix)]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def _load_estimated_times(meta: dict) -> dict[str, float]:
    """Load per-test estimated times from the config meta dict.

    Tests not listed default to 600s when used by _partition_tests.
    """
    return {k: float(v) for k, v in meta.get("estimated_times", {}).items()}


def _load_partition_config(meta: dict) -> dict[str, int]:
    """Load partition configuration from the config meta dict.

    Returns a dict mapping runner keys (e.g. ``a2_x1``) to partition
    counts.  Runner keys not listed default to 1.
    """
    return {k: int(v) for k, v in meta.get("partition", {}).items()}


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


def _build_test_group(
    num_npus: int,
    npu_type: NpuType,
    runner: RunnerInfo,
    tests: list[str],
    partition: str,
) -> dict:
    group: dict = {
        "num_npus": num_npus,
        "npu_type": npu_type.value,
        "runner": runner.label,
        "tests": " ".join(sorted(tests)),
        "partition": partition,
    }
    if runner.image_tag:
        group["image_tag"] = runner.image_tag
    return group


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

        if npu_type == NpuType.A3 and psize > 1:
            # Split A3 tests into SOC-specific (e.g. ascend910_9392) and
            # generic, so they can be routed to different runner pools.
            soc_specific: list[str] = []
            generic: list[str] = []
            for t in tests:
                if _is_soc_specific_test(t):
                    soc_specific.append(t)
                else:
                    generic.append(t)

            # SOC-specific tests are few — no partition needed.
            if soc_specific:
                runner_soc = _find_runner(num_npus, npu_type, runners, label_suffix="")
                if runner_soc is not None:
                    result.append(
                        _build_test_group(
                            num_npus,
                            npu_type,
                            runner_soc,
                            soc_specific,
                            "1-1",
                        )
                    )

            # Generic A3 tests use the full partition config.
            if generic:
                runner_gen = _find_runner(num_npus, npu_type, runners, label_suffix="-")
                if runner_gen is not None:
                    buckets = _partition_tests(sorted(generic), psize, estimated_times)
                    for i, bucket in enumerate(buckets):
                        if not bucket:
                            continue
                        result.append(
                            _build_test_group(
                                num_npus,
                                npu_type,
                                runner_gen,
                                bucket,
                                f"{i + 1}-{psize}",
                            )
                        )
        elif psize > 1:
            buckets = _partition_tests(sorted(tests), psize, estimated_times)
            for i, bucket in enumerate(buckets):
                if not bucket:
                    continue
                result.append(_build_test_group(num_npus, npu_type, runner, bucket, f"{i + 1}-{psize}"))
        else:
            result.append(_build_test_group(num_npus, npu_type, runner, tests, "1-1"))

    if errors:
        details = "".join(errors)
        print(
            f"\nERROR: The following test groups cannot be routed to any runner"
            f" in runner_label.json:\n{details}\n\n"
            "Please fix the directory structure or add the missing runner"
            " to runner_label.json.\n",
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
        "--discover-for-runner",
        type=str,
        metavar="RUNNER_KEY",
        help="Runner key (e.g. a2_x1) to discover tests for. "
        "Scans TEST_PATHS and prints a JSON list of files that would be "
        "routed to the specified runner. Implies --runner-discovery.",
    )
    parser.add_argument(
        "test_paths",
        nargs="*",
        metavar="TEST_PATH",
        help="Test directories or files to scan (only used with --discover-for-runner)",
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
    docs = list(yaml.safe_load_all(args.config.read_text()))
    config = _resolve_config_inheritance(docs[0])
    meta = docs[1] if len(docs) >= 2 and docs[1] else {}
    _load_runner_mapping(meta)
    _load_marker_routing(meta)

    skip_tests: set[str] = set()
    for module in config:
        for s in module.get("skip_tests", []):
            skip_tests.add(s.rstrip("/"))

    if args.discover_for_runner:
        target = _parse_runner_key(args.discover_for_runner)
        discovered = _discover_tests_for_runner(args.test_paths, target)
        print(json.dumps(discovered, separators=(",", ":")))
        return

    if args.explicit_e2e_tests:
        matched_modules: list[str] = []
        all_groups: dict[RunnerKey, list[str]] = defaultdict(list)
        for path in args.explicit_e2e_tests:
            if not _is_e2e_path(path):
                print(
                    f"Warning: Skipping non-e2e path: {path}",
                    file=sys.stderr,
                )
                continue
            _scan_e2e_test_dir(path, all_groups)
    else:
        changed_files = _get_changed_files(args.diff_base) if args.diff_base else args.changed_files
        bisect_tool_scoped_change = _is_bisect_tool_scoped_change(changed_files)
        test_only_change = _is_test_only_change(changed_files)
        if bisect_tool_scoped_change:
            print(
                "Detected bisect tool-scoped change: running only matching tool modules (skipping always-on modules).",
                file=sys.stderr,
            )
        elif test_only_change:
            print(
                "Detected test-only change: running only default_cpu_ut"
                " and the changed test files (skipping source-driven modules).",
                file=sys.stderr,
            )
        if args.run_all_modules:
            matched_modules = [module["name"] for module in config]
        elif bisect_tool_scoped_change:
            matched_modules = _match_modules(changed_files, config, include_always=False)
        elif test_only_change:
            matched_modules = [m["name"] for m in config if m["name"] == DEFAULT_CPU_UT_MODULE]
        else:
            matched_modules = _match_modules(changed_files, config)
        test_dirs, cpu_only_dirs = _collect_test_dirs(matched_modules, config)

        changed_test_files = []
        for f in changed_files:
            if not _is_test_path(f):
                continue
            target = Path(_pytest_node_file_path(f))
            if target.name.startswith("test_") and target.exists():
                changed_test_files.append(f)

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
    estimated_times = _load_estimated_times(meta)
    partition_config = _load_partition_config(meta)
    test_groups = _resolve_to_runners(all_groups, runners, partition_config, estimated_times)

    _write_output(test_groups, matched_modules)


if __name__ == "__main__":
    main()
