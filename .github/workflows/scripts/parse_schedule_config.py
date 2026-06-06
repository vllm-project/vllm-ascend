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
    -> PeriodicCase objects (via _parse_to_case)
    -> framework routers (AccuracyRouter, OpsRouter, ModelSingleNodeRouter, ModelMultiNodeRouter)
    -> framework-specific matrices
    -> GITHUB_OUTPUT

Directory conventions (routing is inferred from path, not config):
  tests/e2e/schedule/model/<Family>/<resource_dir>/*.yaml -> model framework
  tests/e2e/schedule/accuracy/<resource_dir>/*.yaml       -> accuracy framework
  tests/e2e/schedule/ops/<resource_dir>/*.py              -> ops framework
  tests/e2e/schedule/ops/<resource_dir>/                  -> ops framework (directory)

Supported resource directories (English form only; numeric forms fail):
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
  filename stem contains external_dp -> external
  otherwise -> internal

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
    ("card", "a3", 1): 1,
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

# Runners not representable in runner_label.json's (chip, npu_num) space.
_SPECIAL_RUNNERS: dict[tuple[str, int], str] = {
    ("a2", 0): "linux-amd64-cpu-8-hk",
}

# Separator-bounded chip tokens: A2/a2 must not be part of A22B, etc.
# 310p matches the 310 / 310p / v310 token forms used in filenames and dir names.
_CHIP_310P = re.compile(r"(?<![A-Za-z0-9])v?310p?(?![A-Za-z0-9])")
_CHIP_A2 = re.compile(r"(?<![A-Za-z0-9])[Aa]2(?![A-Za-z0-9])")
_CHIP_A3 = re.compile(r"(?<![A-Za-z0-9])[Aa]3(?![A-Za-z0-9])")

# Numeric resource forms that must be rejected.
_NUMERIC_RESOURCE = re.compile(r"(?<![A-Za-z])\d+_(card|node)(?![A-Za-z])")

_SCHEDULE_ROOT = "tests/e2e/schedule"


@dataclass(frozen=True)
class PeriodicCase:
    name: str
    path: str
    framework: str  # "model" | "accuracy" | "ops"
    route: str  # "single_node" | "multi_node" | "accuracy" | "ops"
    chip: str  # "a2" | "a3"
    resource_type: str  # "card" | "node"
    resource_num: int
    resource_dir: str
    runner: str
    family: str | None = None
    multi_node_type: str | None = None  # "internal" | "external" (multi_node only)
    config_path: str | None = None  # single YAML config (model single/multi-node)
    config_paths: list | None = None  # direct accuracy config YAML paths
    tests: str | None = None  # pytest target (ops or accuracy .py)
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
    npu_num = _NPU_NUM.get((resource_type, chip, resource_num), 0)
    key = (chip, npu_num)
    runner = runner_map.get(key) or _SPECIAL_RUNNERS.get(key)
    if runner is None:
        raise ValueError(
            f"No runner for chip={chip!r}, {resource_type}x{resource_num} "
            f"(npu_num={npu_num}). Add an entry to runner_label.json or _SPECIAL_RUNNERS."
        )
    return runner


def _detect_resource(path: str) -> tuple[str, str, int]:
    """Return (resource_dir, resource_type, resource_num). Fail on missing or ambiguous."""
    if _NUMERIC_RESOURCE.search(path):
        raise ValueError(
            f"Numeric resource directory in path {path!r}. "
            f"Use English forms: one_card, two_card, four_card, eight_card, "
            f"one_node, two_node, four_node."
        )
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


def _detect_framework(path: str) -> tuple[str, str | None]:
    """Return (framework, family_or_None). Validate path starts with tests/e2e/schedule/.

    The framework directory is always the 4th path segment:
      tests/e2e/schedule/model/<Family>/...  -> model    (family = 5th segment)
      tests/e2e/schedule/accuracy/...        -> accuracy
      tests/e2e/schedule/ops/...             -> ops
    """
    norm = path.replace("\\", "/").rstrip("/")
    if norm.startswith("tests/e2e/accuracy/"):
        raise ValueError(
            f"Invalid path {path!r}: tests/e2e/accuracy/ is not supported. Use tests/e2e/schedule/accuracy/ instead."
        )
    if not norm.startswith(_SCHEDULE_ROOT + "/"):
        raise ValueError(f"Path {path!r} must start with {_SCHEDULE_ROOT}/.")
    parts = Path(norm).parts
    if len(parts) < 4:
        raise ValueError(f"Path {path!r} is too short; expected tests/e2e/schedule/<model|accuracy|ops>/...")
    framework = parts[3]
    if framework == "accuracy":
        return "accuracy", None
    if framework == "ops":
        return "ops", None
    if framework == "model":
        if len(parts) < 5 or parts[4] in RESOURCE_DIRS:
            raise ValueError(
                f"Model path {path!r} must include a family: tests/e2e/schedule/model/<Family>/<resource_dir>/..."
            )
        return "model", parts[4]
    raise ValueError(f"Unknown framework directory {framework!r} in {path!r}; expected one of: model, accuracy, ops.")


def _detect_route(framework: str, resource_type: str, resource_num: int) -> str:
    if framework == "accuracy":
        if resource_type != "card":
            raise ValueError(
                "Accuracy framework only supports card resources. "
                "Found node resource. Move accuracy node cases to a separate framework."
            )
        return "accuracy"
    if framework == "ops":
        return "ops"
    # model
    if resource_type == "card" or resource_num == 1:
        return "single_node"
    return "multi_node"


def _detect_chip(path: str) -> str:
    norm = path.replace("\\", "/")
    if _CHIP_310P.search(norm):
        return "310p"
    if _CHIP_A2.search(norm):
        return "a2"
    if _CHIP_A3.search(norm):
        return "a3"
    return "a3"


def _detect_multi_node_type(path: str) -> str:
    stem = Path(path.replace("\\", "/")).stem
    return "external" if "external_dp" in stem else "internal"


def _derive_name(path: str) -> str:
    stem = Path(path.replace("\\", "/").rstrip("/")).stem
    return re.sub(r"[_.]", "-", stem).lower()


def _validate_accuracy_config(path: str) -> None:
    """Ensure an accuracy YAML is a real config (a dict), not an old model-list group.

    Validation is skipped when the file is absent (the parser may run before the
    referenced path is materialized); when present, a list-based group YAML is
    rejected so old files cannot be silently treated as real configs.
    """
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(
            f"Accuracy config {path!r} must be a YAML dict. Old model-list group YAML is no longer supported."
        )


def _parse_to_case(raw: Any, runner_map: dict[tuple[str, int], str]) -> PeriodicCase:
    """Parse a single schedule_config.yaml files entry into a PeriodicCase."""
    path = str(raw).strip().replace("\\", "/").rstrip("/")

    resource_dir, resource_type, resource_num = _detect_resource(path)
    framework, family = _detect_framework(path)
    route = _detect_route(framework, resource_type, resource_num)
    chip = _detect_chip(path)
    runner = _resolve_runner(chip, resource_type, resource_num, runner_map)
    name = _derive_name(path)

    config_path: str | None = None
    config_paths: list | None = None
    tests_path: str | None = None
    multi_node_type: str | None = None
    size: int | None = None

    if route == "multi_node":
        multi_node_type = _detect_multi_node_type(path)
        size = resource_num
        config_path = path
    elif route == "single_node":
        config_path = path
    elif route == "accuracy":
        if path.endswith((".yaml", ".yml")):
            _validate_accuracy_config(path)
            config_paths = [path]
        else:
            raise ValueError(f"Accuracy entries must be YAML configs: {path}")
    elif route == "ops":
        tests_path = path

    return PeriodicCase(
        name=name,
        path=path,
        framework=framework,
        route=route,
        chip=chip,
        resource_type=resource_type,
        resource_num=resource_num,
        resource_dir=resource_dir,
        runner=runner,
        family=family,
        multi_node_type=multi_node_type,
        config_path=config_path,
        config_paths=config_paths,
        tests=tests_path,
        size=size,
    )


# ---------------------------------------------------------------------------
# Directory entry expansion (a config entry may be a directory)
# ---------------------------------------------------------------------------


def _is_directory_entry(raw: Any, path: str) -> bool:
    """Heuristic: an entry is a directory if it ends with '/', exists as a dir,
    or carries no file extension."""
    if str(raw).rstrip().endswith("/"):
        return True
    if Path(path).is_dir():
        return True
    return Path(path).suffix == ""


def _list_dir_files(dir_path: str, framework: str) -> list[str]:
    """Recursively list the routable files inside a directory entry."""
    p = Path(dir_path)
    patterns = ["test_*.py"] if framework == "ops" else ["*.yaml", "*.yml"]
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


def _expand_directory(dir_path: str, runner_map: dict[tuple[str, int], str]) -> list[PeriodicCase]:
    """Expand a directory entry into PeriodicCase objects (per the chosen semantics).

    ops      -> group files by detected chip; one case per chip, tests = file list.
    accuracy -> group files by detected chip; one case per chip, config_paths = list.
    model    -> one case per file (each YAML is its own config_path).
    """
    resource_dir, resource_type, resource_num = _detect_resource(dir_path)
    framework, _ = _detect_framework(dir_path)
    files = _list_dir_files(dir_path, framework)
    if not files:
        raise ValueError(f"Directory entry {dir_path!r} contains no routable files to expand.")

    if framework == "accuracy":
        # accuracy: validate each config, group by chip into config_paths lists.
        for f in files:
            _validate_accuracy_config(f)
        cases: list[PeriodicCase] = []
        for chip in sorted(_group_by_chip(files)):
            group_files = sorted(_group_by_chip(files)[chip])
            cases.append(
                PeriodicCase(
                    name=f"{resource_dir}-{chip}",
                    path=dir_path,
                    framework="accuracy",
                    route="accuracy",
                    chip=chip,
                    resource_type=resource_type,
                    resource_num=resource_num,
                    resource_dir=resource_dir,
                    runner=_resolve_runner(chip, resource_type, resource_num, runner_map),
                    config_paths=group_files,
                )
            )
        return cases

    if framework != "ops":
        # model directories: each file routes independently.
        return [_parse_to_case(f, runner_map) for f in files]

    # ops: group by chip so each group runs on the correct runner.
    groups = _group_by_chip(files)

    base = _derive_name(dir_path)
    cases = []
    for chip in sorted(groups):
        group_files = sorted(groups[chip])
        cases.append(
            PeriodicCase(
                name=f"{base}-{chip}",
                path=dir_path,
                framework="ops",
                route="ops",
                chip=chip,
                resource_type=resource_type,
                resource_num=resource_num,
                resource_dir=resource_dir,
                runner=_resolve_runner(chip, resource_type, resource_num, runner_map),
                tests=" ".join(group_files),
            )
        )
    return cases


def _parse_entry(raw: Any, runner_map: dict[tuple[str, int], str]) -> list[PeriodicCase]:
    """Parse one schedule_config.yaml entry into one or more PeriodicCase objects.

    File entries -> a single case. Directory entries are expanded before routing.
    """
    path = str(raw).strip().replace("\\", "/").rstrip("/")
    if _is_directory_entry(raw, path):
        return _expand_directory(path, runner_map)
    return [_parse_to_case(path, runner_map)]


# ---------------------------------------------------------------------------
# Framework routers
# ---------------------------------------------------------------------------


class BaseRouter:
    name: str
    output_name: str

    def match(self, case: PeriodicCase) -> bool:
        raise NotImplementedError

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        raise NotImplementedError


class AccuracyRouter(BaseRouter):
    name = "accuracy"
    output_name = "accuracy_matrix"

    def match(self, case: PeriodicCase) -> bool:
        return case.framework == "accuracy"

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        return {
            "name": case.name,
            "chip": case.chip,
            "runner": case.runner,
            "config_paths": case.config_paths or ([case.config_path] if case.config_path else []),
        }


class OpsRouter(BaseRouter):
    name = "ops"
    output_name = "ops_matrix"

    def match(self, case: PeriodicCase) -> bool:
        return case.framework == "ops"

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        return {
            "name": case.name,
            "chip": case.chip,
            "runner": case.runner,
            "tests": case.tests or "",
        }


class ModelSingleNodeRouter(BaseRouter):
    name = "single_node"
    output_name = "single_node_matrix"

    def match(self, case: PeriodicCase) -> bool:
        return case.framework == "model" and case.route == "single_node"

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        return {
            "name": case.name,
            "chip": case.chip,
            "runner": case.runner,
            "config_path": case.config_path or "",
            "tests": "",
            "extra_components": False,
        }


class ModelMultiNodeRouter(BaseRouter):
    name = "multi_node"
    output_name = "multi_node_matrix"

    def match(self, case: PeriodicCase) -> bool:
        return case.framework == "model" and case.route == "multi_node"

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        return {
            "name": case.name,
            "chip": case.chip,
            "runner": case.runner,
            "config_path": case.config_path or "",
            "multi_node_type": case.multi_node_type or "internal",
            "extra_components": False,
            "size": case.size or case.resource_num,
        }


ROUTERS: list[BaseRouter] = [
    AccuracyRouter(),
    OpsRouter(),
    ModelSingleNodeRouter(),
    ModelMultiNodeRouter(),
]


# ---------------------------------------------------------------------------
# Schedule selection, filtering, deduplication
# ---------------------------------------------------------------------------


def _select_schedules(config: dict, event_name: str, cron: str, schedule_name: str) -> list[dict]:
    """Return schedule sections matching the current trigger."""
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


def _dedupe_cases(cases: list[PeriodicCase]) -> list[PeriodicCase]:
    """Remove duplicates by name (keep first occurrence)."""
    seen: set[str] = set()
    result = []
    for c in cases:
        if c.name not in seen:
            seen.add(c.name)
            result.append(c)
    return result


def _matches_filter(case: PeriodicCase, test_filter: str) -> bool:
    if test_filter == "all":
        return True
    path = case.config_path or case.tests or case.path
    name = case.name
    filename = Path(path).name if path else ""
    stem = Path(path).stem if path else ""
    # Priority: full path > filename > stem > path segment > name/path substring
    for target in [path, filename, stem]:
        if target and target == test_filter:
            return True
    if path:
        if any(part == test_filter for part in Path(path.replace("\\", "/")).parts):
            return True
    return any(target and test_filter in target for target in [name, path])


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

    schedules = _select_schedules(config, args.event_name, args.cron, args.schedule_name)
    if not schedules:
        print(
            f"No schedules matched event={args.event_name!r} cron={args.cron!r} schedule_name={args.schedule_name!r}",
            file=sys.stderr,
        )

    # Parse every entry into one or more PeriodicCase objects
    # (directory entries are expanded before routing).
    all_cases: list[PeriodicCase] = []
    errors: list[str] = []
    for schedule in schedules:
        for raw in schedule.get("files", []):
            try:
                all_cases.extend(_parse_entry(raw, runner_map))
            except Exception as exc:
                errors.append(f"  {raw!r}: {exc}")

    if errors:
        print("Errors parsing schedule entries:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    all_cases = _dedupe_cases(all_cases)

    test_filter = args.test_filter.strip()
    if test_filter:
        all_cases = [c for c in all_cases if _matches_filter(c, test_filter)]

    # Route through framework routers
    matrices: dict[str, list[dict]] = {r.output_name: [] for r in ROUTERS}
    unmatched: list[PeriodicCase] = []
    for case in all_cases:
        matched = False
        for router in ROUTERS:
            if router.match(case):
                matrices[router.output_name].append(router.to_matrix_item(case))
                matched = True
                break
        if not matched:
            unmatched.append(case)

    if unmatched:
        for c in unmatched:
            print(
                f"ERROR: no router matched case {c.name!r} (framework={c.framework!r}, route={c.route!r})",
                file=sys.stderr,
            )
        sys.exit(1)

    # Sort multi-node: largest resource consumer first (4-node before 2-node)
    matrices["multi_node_matrix"].sort(key=lambda e: -e.get("size", 0))

    # Image build targets derived from chip set of selected cases
    image_targets = sorted({c.chip for c in all_cases})

    # Human-readable summary
    summary_lines = ["=== Selected test cases ==="]
    for c in all_cases:
        loc = c.config_path or c.tests or c.path
        summary_lines.append(f"  [{c.framework:8s}] [{c.route:11s}] [{c.chip}] [{c.runner:30s}] {c.name} ({loc})")
    summary_lines.append(
        f"\nTotals: "
        f"{len(matrices['single_node_matrix'])} single-node, "
        f"{len(matrices['multi_node_matrix'])} multi-node, "
        f"{len(matrices['accuracy_matrix'])} accuracy, "
        f"{len(matrices['ops_matrix'])} ops"
    )
    summary = "\n".join(summary_lines)
    print(summary, file=sys.stderr)

    output_path = os.environ.get("GITHUB_OUTPUT", "")
    lines = [
        f"single_node_matrix={json.dumps(matrices['single_node_matrix'])}",
        f"multi_node_matrix={json.dumps(matrices['multi_node_matrix'])}",
        f"accuracy_matrix={json.dumps(matrices['accuracy_matrix'])}",
        f"ops_matrix={json.dumps(matrices['ops_matrix'])}",
        f"image_build_targets={json.dumps(image_targets)}",
        f"selected_cases_summary={json.dumps(summary)}",
    ]

    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n=== Outputs ===")
        for output_name in ("single_node_matrix", "multi_node_matrix", "accuracy_matrix", "ops_matrix"):
            items = matrices[output_name]
            print(f"\n{output_name} ({len(items)} entries):")
            print(json.dumps(items, indent=2))
        print(f"\nimage_build_targets: {image_targets}")


if __name__ == "__main__":
    main()
