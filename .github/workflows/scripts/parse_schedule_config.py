#!/usr/bin/env python3
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
"""Parse schedule_config.yaml and output test matrices for GitHub Actions.

Architecture:
  raw file paths from schedule_config.yaml
    -> framework classes (ModelFramework, AccuracyFramework, OpsFramework)
    -> PeriodicCase objects
    -> framework-specific matrices
    -> GITHUB_OUTPUT

Directory conventions (routing is inferred from path, not config):
  tests/e2e/schedule/model/.../<resource_dir>/*.yaml      -> model framework
  tests/e2e/schedule/accuracy/<resource_dir>/*.yaml       -> accuracy framework
  tests/e2e/schedule/ops/<resource_dir>/*.py              -> ops framework
  tests/e2e/schedule/ops/<resource_dir>/                  -> ops framework (directory)

Supported resource directories:
  one_card, two_card, four_card, eight_card -> card resources
  one_node, two_node, four_node             -> node resources

Chip detection (separator-bounded token in path or filename):
  contains 310/310p/v310 -> 310p
  contains a2/A2         -> a2
  contains a3/A3         -> a3
  default               -> a3

Route rules:
  model + card or one_node -> single_node
  model + two_node/four_node -> multi_node
  accuracy + card -> accuracy (node resources not supported)
  ops + any -> ops

Multi-node type (model multi-node only):
  filename stem contains external_dp -> external_dp
  otherwise -> internal_dp

Usage:
    python parse_schedule_config.py \\
        --config .github/workflows/scripts/schedule_config.yaml \\
        --runner-label .github/workflows/scripts/runner_label.json \\
        --event-name schedule \\
        --cron "45 15 * * *" \\
        --test-filter all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import yaml

RESOURCE_DIRS: dict[str, tuple[str, int]] = {
    "one_card": ("card", 1),
    "two_card": ("card", 2),
    "four_card": ("card", 4),
    "eight_card": ("card", 8),
    "one_node": ("node", 1),
    "two_node": ("node", 2),
    "four_node": ("node", 4),
}

# (resource_type, chip, resource_num) -> npu_num for runner_label.json lookup.
# npu_num=0 means LWS orchestration runner (no NPUs on the control node).
_NPU_NUM: dict[tuple[str, str, int], int] = {
    ("card", "a3", 1): 2,
    ("card", "a3", 2): 2,
    ("card", "a3", 4): 4,
    ("card", "a3", 8): 8,
    ("node", "a3", 1): 16,
    ("node", "a3", 2): 0,
    ("node", "a3", 4): 0,
    ("card", "a2", 1): 1,
    ("card", "a2", 2): 2,
    ("card", "a2", 4): 4,
    ("card", "a2", 8): 8,
    ("node", "a2", 1): 8,
    ("node", "a2", 2): 0,
    ("node", "a2", 4): 0,
    ("card", "310p", 1): 1,
    ("card", "310p", 2): 2,
    ("card", "310p", 4): 4,
}

# Separator-bounded chip tokens: A2/a2 must not be part of A22B, etc.
# 310p matches the 310 / 310p / v310 token forms used in filenames and dir names.
_CHIP_310P = re.compile(r"(?<![A-Za-z0-9])v?310p?(?![A-Za-z0-9])")
_CHIP_A2 = re.compile(r"(?<![A-Za-z0-9])[Aa]2(?![A-Za-z0-9])")
_CHIP_A3 = re.compile(r"(?<![A-Za-z0-9])[Aa]3(?![A-Za-z0-9])")

_SCHEDULE_ROOT = "tests/e2e/schedule"

# These names are part of the GitHub Actions contract. Keep them stable unless
# every consuming workflow is migrated at the same time.
_MATRIX_OUTPUT_NAMES = (
    "single_node_matrix",
    "multi_node_matrix",
    "accuracy_matrix",
    "ops_matrix",
)


@dataclass(frozen=True)
class PeriodicCase:
    """Normalized test case used between framework parsing and matrix output.

    The source schedule only contains paths. Frameworks expand those paths into
    this shared shape so the main flow can dedupe, filter, summarize, and merge
    cases without knowing model/accuracy/ops-specific details.
    """

    name: str
    path: str
    framework: str  # "model" | "accuracy" | "ops"
    route: str  # "single_node" | "multi_node" | "accuracy" | "ops"
    chip: str  # "a2" | "a3"
    resource_type: str  # "card" | "node"
    resource_num: int
    resource_dir: str
    runner: str
    multi_node_type: str | None = None  # "internal_dp" | "external_dp" (multi_node only)
    # Framework-specific config/test payload:
    #   model    -> single YAML path
    #   accuracy -> YAML path list passed as config_paths
    #   ops      -> pytest target path or grouped path list
    case_path: list[str] | str = ""
    size: int | None = None  # node count for multi-node


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _load_runner_map(runner_label_path: Path) -> dict[tuple[str, int], str]:
    """Build (chip, npu_num) -> runner_label reverse map from runner_label.json."""
    with open(runner_label_path, encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
    return {(info["chip"], info["npu_num"]): label for label, info in data.items()}


def _resolve_runner(
    chip: str,
    resource_type: str,
    resource_num: int,
    runner_map: dict[tuple[str, int], str],
) -> str:
    """Resolve a runner label from chip/resource metadata.

    The schedule config intentionally does not carry runner labels. It only
    encodes topology through the path, then this helper maps that topology to
    runner_label.json's (chip, npu_num) space.
    """
    npu_num = _NPU_NUM.get((resource_type, chip, resource_num), 0)
    key = (chip, npu_num)
    runner = runner_map.get(key)
    if runner is None:
        raise ValueError(
            f"No runner for chip={chip!r}, {resource_type}x{resource_num} "
            f"(npu_num={npu_num}). Add an entry to runner_label.json."
        )
    return runner


def _detect_resource(path: str) -> tuple[str, str, int]:
    """Return (resource_dir, resource_type, resource_num).

    Every routable path must include exactly one supported resource directory.
    This keeps ambiguous paths from silently choosing the wrong matrix.
    """
    parts = Path(path.replace("\\", "/")).parts
    matches = [p for p in parts if p in RESOURCE_DIRS]
    if not matches:
        raise ValueError(f"No resource directory in path {path!r}. Expected one of: {', '.join(RESOURCE_DIRS)}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple resource directories in path {path!r}: {matches}. Each path must contain exactly one."
        )
    resource_dir = matches[0]
    resource_type, resource_num = RESOURCE_DIRS[resource_dir]
    return resource_dir, resource_type, resource_num


def _detect_chip(path: str) -> str:
    """Infer chip from separator-bounded path tokens, defaulting to a3."""
    norm = path.replace("\\", "/")
    if _CHIP_310P.search(norm):
        return "310p"
    if _CHIP_A2.search(norm):
        return "a2"
    if _CHIP_A3.search(norm):
        return "a3"
    return "a3"


def _detect_multi_node_type(path: str) -> str:
    """Infer the multi-node launch mode from the model config filename."""
    stem = Path(path.replace("\\", "/")).stem
    return "external_dp" if "external_dp" in stem else "internal_dp"


def _derive_name(path: str) -> str:
    """Use the path stem as the stable case name for dedupe and summaries."""
    return Path(path.replace("\\", "/").rstrip("/")).stem


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def _normalize_path(raw: Any) -> str:
    """Normalize schedule entries before matching or expanding frameworks."""
    return str(raw).strip().replace("\\", "/").rstrip("/")


def _is_directory_entry(raw: Any, path: str) -> bool:
    """Heuristic: an entry is a directory if it ends with '/', exists as a dir,
    or carries no file extension."""
    if str(raw).rstrip().endswith("/"):
        return True
    if Path(path).is_dir():
        return True
    return Path(path).suffix == ""


def _list_dir_files(dir_path: str, patterns: list[str]) -> list[str]:
    """Recursively list files matching framework-provided patterns."""
    p = Path(dir_path)
    files: set[str] = set()
    for pat in patterns:
        files.update(str(f).replace("\\", "/") for f in p.rglob(pat))
    return sorted(files)


def _group_by_chip(files: list[str]) -> dict[str, list[str]]:
    """Group files by their detected chip (deterministic ordering preserved)."""
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(_detect_chip(f), []).append(f)
    return groups


# ---------------------------------------------------------------------------
# Frameworks
# ---------------------------------------------------------------------------


class BaseFramework:
    """Interface for framework-specific path parsing and matrix conversion."""

    name: str
    output_names: tuple[str, ...]

    def __init__(self, runner_map: dict[tuple[str, int], str]):
        self.runner_map = runner_map

    def match(self, path: str) -> bool:
        """Return True when this framework owns the raw schedule path."""
        raise NotImplementedError

    def expand(self, raw: Any) -> list[PeriodicCase]:
        """Convert a raw file or directory entry into normalized cases."""
        raise NotImplementedError

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        """Convert this framework's cases into GitHub Actions matrix items."""
        raise NotImplementedError


class ModelFramework(BaseFramework):
    """Handle model YAML configs and split them into single/multi-node matrices."""

    name = "model"
    output_names = ("single_node_matrix", "multi_node_matrix")

    def match(self, path: str) -> bool:
        return _normalize_path(path).startswith(f"{_SCHEDULE_ROOT}/model/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = _normalize_path(raw)
        if _is_directory_entry(raw, path):
            # Directory entries still need an explicit resource segment. Without
            # it, nested YAMLs from different topologies could be mixed together.
            _detect_resource(path)
            files = _list_dir_files(path, ["*.yaml", "*.yml"])
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no routable files.")
            return [self._case_from_file(f) for f in files]
        return [self._case_from_file(path)]

    def _case_from_file(self, path: str) -> PeriodicCase:
        """Parse one model YAML into a case and decide its route."""
        resource_dir, resource_type, resource_num = _detect_resource(path)
        chip = _detect_chip(path)

        if resource_type == "card" or resource_num == 1:
            route = "single_node"
            runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)
            multi_node_type = None
            size = None
        else:
            route = "multi_node"
            runner = ""
            multi_node_type = _detect_multi_node_type(path)
            size = resource_num

        return PeriodicCase(
            name=_derive_name(path),
            path=path,
            framework=self.name,
            route=route,
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            multi_node_type=multi_node_type,
            case_path=path,
            size=size,
        )

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        """Build model matrices while preserving workflow output shape."""
        single_node = []
        multi_node = []
        for case in cases:
            if case.route == "single_node":
                single_node.append(
                    {
                        "name": case.name,
                        "chip": case.chip,
                        "runner": case.runner,
                        "config_path": case.case_path,
                        "tests": "",
                        "extra_components": False,
                    }
                )
            elif case.route == "multi_node":
                multi_node.append(
                    {
                        "name": case.name,
                        "chip": case.chip,
                        "config_path": case.case_path,
                        "multi_node_type": case.multi_node_type or "internal_dp",
                        "extra_components": False,
                        "size": case.size or case.resource_num,
                    }
                )
            else:
                raise ValueError(f"Unknown model route: {case.route}")

        multi_node.sort(key=lambda e: -e.get("size", 0))
        return {
            "single_node_matrix": single_node,
            "multi_node_matrix": multi_node,
        }


class AccuracyFramework(BaseFramework):
    """Handle accuracy YAML configs, grouping directory entries by chip."""

    name = "accuracy"
    output_names = ("accuracy_matrix",)

    def match(self, path: str) -> bool:
        return _normalize_path(path).startswith(f"{_SCHEDULE_ROOT}/accuracy/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = _normalize_path(raw)
        if _is_directory_entry(raw, path):
            # Accuracy directories become one job per chip, so the directory
            # itself must identify a single resource size.
            _detect_resource(path)
            files = _list_dir_files(path, ["*.yaml", "*.yml"])
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no routable files.")
            return self._cases_from_directory(path, files)
        if not path.endswith((".yaml", ".yml")):
            raise ValueError(f"Accuracy entries must be YAML configs: {path}")
        return [self._case_from_files(path, [path])]

    def _case_from_files(self, path: str, files: list[str]) -> PeriodicCase:
        """Create one accuracy case from one or more YAML config paths."""
        resource_dir, resource_type, resource_num = _detect_resource(path)
        if resource_type != "card":
            raise ValueError("Accuracy framework only supports card resources.")

        chip = _detect_chip(path)
        runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)
        return PeriodicCase(
            name=_derive_name(path),
            path=path,
            framework=self.name,
            route="accuracy",
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            case_path=files,
        )

    def _cases_from_directory(self, dir_path: str, files: list[str]) -> list[PeriodicCase]:
        """Bundle directory YAMLs by chip so each group uses the right runner."""
        resource_dir, resource_type, resource_num = _detect_resource(dir_path)
        if resource_type != "card":
            raise ValueError("Accuracy framework only supports card resources.")

        cases = []
        groups = _group_by_chip(files)
        for chip in sorted(groups):
            group_files = sorted(groups[chip])
            cases.append(
                PeriodicCase(
                    name=f"{resource_dir}-{chip}",
                    path=dir_path,
                    framework=self.name,
                    route="accuracy",
                    chip=chip,
                    resource_type=resource_type,
                    resource_num=resource_num,
                    resource_dir=resource_dir,
                    runner=_resolve_runner(chip, resource_type, resource_num, self.runner_map),
                    case_path=group_files,
                )
            )
        return cases

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        """Expose accuracy payloads as config_paths for the workflow."""
        return {
            "accuracy_matrix": [
                {
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "config_paths": case.case_path,
                }
                for case in cases
            ]
        }


class OpsFramework(BaseFramework):
    """Handle pytest ops targets, grouping directory entries by chip."""

    name = "ops"
    output_names = ("ops_matrix",)

    def match(self, path: str) -> bool:
        return _normalize_path(path).startswith(f"{_SCHEDULE_ROOT}/ops/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = _normalize_path(raw)
        if _is_directory_entry(raw, path):
            # Ops directories can contain chip-specific files. Group after
            # discovery so 310p/a2/a3 tests land on matching runners.
            _detect_resource(path)
            files = _list_dir_files(path, ["test_*.py"])
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no routable files.")
            return self._cases_from_directory(path, files)
        return [self._case_from_file(path)]

    def _case_from_file(self, path: str) -> PeriodicCase:
        """Create one ops case from a direct pytest target path."""
        resource_dir, resource_type, resource_num = _detect_resource(path)
        chip = _detect_chip(path)
        runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)
        return PeriodicCase(
            name=_derive_name(path),
            path=path,
            framework=self.name,
            route="ops",
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            case_path=path,
        )

    def _cases_from_directory(self, dir_path: str, files: list[str]) -> list[PeriodicCase]:
        """Bundle discovered pytest files by chip for per-runner jobs."""
        resource_dir, resource_type, resource_num = _detect_resource(dir_path)
        base_name = _derive_name(dir_path)
        cases = []
        groups = _group_by_chip(files)
        for chip in sorted(groups):
            group_files = sorted(groups[chip])
            cases.append(
                PeriodicCase(
                    name=f"{base_name}-{chip}",
                    path=dir_path,
                    framework=self.name,
                    route="ops",
                    chip=chip,
                    resource_type=resource_type,
                    resource_num=resource_num,
                    resource_dir=resource_dir,
                    runner=_resolve_runner(chip, resource_type, resource_num, self.runner_map),
                    case_path=group_files,
                )
            )
        return cases

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        """Expose ops payloads as space-separated pytest targets."""
        return {
            "ops_matrix": [
                {
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "tests": " ".join(case.case_path) if isinstance(case.case_path, list) else case.case_path,
                }
                for case in cases
            ]
        }


def _build_frameworks(runner_map: dict[tuple[str, int], str]) -> list[BaseFramework]:
    """Register supported frameworks in matching order."""
    return [
        ModelFramework(runner_map),
        AccuracyFramework(runner_map),
        OpsFramework(runner_map),
    ]


def _find_framework(path: str, frameworks: list[BaseFramework]) -> BaseFramework:
    """Find the single framework responsible for a raw schedule path."""
    matched = [fw for fw in frameworks if fw.match(path)]
    if not matched:
        raise ValueError(f"No framework matched path {path!r}.")
    if len(matched) > 1:
        names = [fw.name for fw in matched]
        raise ValueError(f"Multiple frameworks matched path {path!r}: {names}")
    return matched[0]


# ---------------------------------------------------------------------------
# Schedule selection, filtering, deduplication
# ---------------------------------------------------------------------------


def _select_schedules(config: dict, event_name: str, cron: str, schedule_name: str) -> list[dict]:
    """Return schedule sections matching the current GitHub event."""
    selected = []
    for schedule in config.get("periodic_tests", []):
        sched_cron = schedule.get("cron", "")
        sched_name = schedule.get("name", "")
        if event_name == "schedule" and cron:
            if sched_cron == cron:
                selected.append(schedule)
        elif event_name == "workflow_dispatch":
            if schedule_name and sched_name == schedule_name:
                selected.append(schedule)
            elif not schedule_name or schedule_name == "manual":
                if schedule.get("files"):
                    selected.append(schedule)
    return selected


def _dedupe_cases(cases: list[PeriodicCase]) -> tuple[list[PeriodicCase], dict[str, list[PeriodicCase]]]:
    """Remove duplicates by name and record duplicate groups for reporting."""
    seen: dict[str, PeriodicCase] = {}
    duplicates: dict[str, list[PeriodicCase]] = {}
    result = []
    for c in cases:
        first_case = seen.get(c.name)
        if first_case is None:
            seen[c.name] = c
            result.append(c)
        else:
            duplicates.setdefault(c.name, [first_case]).append(c)
    return result, duplicates


def _split_test_filters(test_filter: str) -> list[str]:
    """Normalize comma-separated --test-filter input into individual filters."""
    filters = [item.strip() for item in test_filter.split(",") if item.strip()]
    return filters or ["all"]


def _matches_filter(case: PeriodicCase, test_filter: str) -> bool:
    """Match one or more filters against path, filename, stem, segment, or name."""
    filters = _split_test_filters(test_filter)
    if len(filters) > 1:
        return any(_matches_filter(case, item) for item in filters)

    test_filter = filters[0]
    if test_filter == "all":
        return True
    paths = case.case_path if isinstance(case.case_path, list) else [case.case_path or case.path]
    name = case.name
    # Priority: full path > filename > stem > path segment > name/path substring
    for path in paths:
        filename = Path(path).name if path else ""
        stem = Path(path).stem if path else ""
        for target in [path, filename, stem]:
            if target and target == test_filter:
                return True
        if any(part == test_filter for part in Path(path.replace("\\", "/")).parts):
            return True
    return test_filter in name or any(path and test_filter in path for path in paths)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _build_summary(
    all_cases: list[PeriodicCase],
    outputs: dict[str, list[dict]],
    duplicate_cases: dict[str, list[PeriodicCase]],
) -> str:
    """Build the human-readable summary emitted to stderr and GITHUB_OUTPUT."""
    summary_lines = ["=== Selected test cases ==="]
    for c in all_cases:
        loc = ", ".join(c.case_path) if isinstance(c.case_path, list) else c.case_path
        runner = c.runner or "workflow-default"
        summary_lines.append(f"  [{c.framework:8s}] [{c.route:11s}] [{c.chip}] [{runner:30s}] {c.name} ({loc})")
    summary_lines.append(
        f"\nTotals: "
        f"{len(outputs.get('single_node_matrix', []))} single-node, "
        f"{len(outputs.get('multi_node_matrix', []))} multi-node, "
        f"{len(outputs.get('accuracy_matrix', []))} accuracy, "
        f"{len(outputs.get('ops_matrix', []))} ops"
    )
    if duplicate_cases:
        summary_lines.append("\nWARNING: duplicate test case names detected; kept the first occurrence:")
        for name in sorted(duplicate_cases):
            cases = duplicate_cases[name]
            summary_lines.append(f"  {name}:")
            summary_lines.append(f"    kept: {cases[0].path}")
            for case in cases[1:]:
                summary_lines.append(f"    duplicate: {case.path}")
    return "\n".join(summary_lines)


def _write_outputs(
    outputs: dict[str, list[dict]],
    image_targets: list[str],
    summary: str,
) -> None:
    """Write GitHub Actions outputs, or print debug output for local runs."""
    print(summary, file=sys.stderr)

    lines = [
        f"single_node_matrix={json.dumps(outputs.get('single_node_matrix', []))}",
        f"multi_node_matrix={json.dumps(outputs.get('multi_node_matrix', []))}",
        f"accuracy_matrix={json.dumps(outputs.get('accuracy_matrix', []))}",
        f"ops_matrix={json.dumps(outputs.get('ops_matrix', []))}",
        f"image_build_targets={json.dumps(image_targets)}",
        f"selected_cases_summary={json.dumps(summary)}",
    ]

    output_path = os.environ.get("GITHUB_OUTPUT", "")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n=== Outputs ===")
        for output_name in _MATRIX_OUTPUT_NAMES:
            items = outputs.get(output_name, [])
            print(f"\n{output_name} ({len(items)} entries):")
            print(json.dumps(items, indent=2))
        print(f"\nimage_build_targets: {image_targets}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to schedule_config.yaml")
    parser.add_argument("--runner-label", help="Path to runner_label.json (default: same dir as config)")
    parser.add_argument("--event-name", default="workflow_dispatch")
    parser.add_argument("--cron", default="")
    parser.add_argument("--schedule-name", default="")
    parser.add_argument("--test-filter", default="all")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runner_label_path = Path(args.runner_label) if args.runner_label else Path(args.config).parent / "runner_label.json"
    runner_map = _load_runner_map(runner_label_path)
    frameworks = _build_frameworks(runner_map)

    schedules = _select_schedules(config, args.event_name, args.cron, args.schedule_name)
    if not schedules:
        print(
            f"No schedules matched event={args.event_name!r} cron={args.cron!r} schedule_name={args.schedule_name!r}",
            file=sys.stderr,
        )

    all_cases: list[PeriodicCase] = []
    errors: list[str] = []
    for schedule in schedules:
        for raw in schedule.get("files", []):
            path = _normalize_path(raw)
            try:
                # Dispatch is path-prefix based. Frameworks own all parsing and
                # expansion after this point, including directory semantics.
                framework = _find_framework(path, frameworks)
                all_cases.extend(framework.expand(raw))
            except Exception as exc:
                errors.append(f"  {raw!r}: {exc}")

    if errors:
        print("Errors parsing schedule entries:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    all_cases, duplicate_cases = _dedupe_cases(all_cases)

    test_filter = args.test_filter.strip()
    if test_filter:
        all_cases = [c for c in all_cases if _matches_filter(c, test_filter)]

    # Keep main framework-agnostic: split by framework name, then let each
    # framework produce its own workflow-compatible matrix payloads.
    cases_by_framework: dict[str, list[PeriodicCase]] = {fw.name: [] for fw in frameworks}
    for case in all_cases:
        cases_by_framework[case.framework].append(case)

    outputs: dict[str, list[dict]] = {name: [] for name in _MATRIX_OUTPUT_NAMES}
    for fw in frameworks:
        grouped = fw.group(cases_by_framework[fw.name])
        for output_name, items in grouped.items():
            # Guard the public output contract so a framework cannot introduce a
            # new output name without an explicit workflow update.
            if output_name not in outputs:
                raise ValueError(f"Framework {fw.name!r} returned unknown output {output_name!r}.")
            outputs[output_name].extend(items)

    image_targets = sorted({c.chip for c in all_cases})
    summary = _build_summary(all_cases, outputs, duplicate_cases)
    _write_outputs(outputs, image_targets, summary)


if __name__ == "__main__":
    main()
