#!/usr/bin/env python3
"""Check and backfill per-test coverage data for full coverage runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def coverage_key(target: str) -> str:
    """Return the output directory name used by run_selected_tests.sh."""
    basename = target[:-3] if target.endswith(".py") else target
    return basename.replace("\\", "/").replace("/", "__").replace("::", "--")


def expected_coverage_keys(test_groups: list[dict[str, Any]]) -> set[str]:
    """Collect the coverage output keys expected from the selected test groups."""
    keys: set[str] = set()
    for group in test_groups:
        if group.get("npu_type") == "cpu":
            if group.get("tests", "").split():
                keys.add("cpu-ut")
            continue
        keys.update(coverage_key(target) for target in group.get("tests", "").split())
    return keys


def _coverage_dirs(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not root.exists():
        return result
    for covdata_dir in root.rglob("covdata"):
        if not covdata_dir.is_dir() or not any(path.is_file() for path in covdata_dir.rglob("*")):
            continue
        result.setdefault(covdata_dir.parent.name, []).append(covdata_dir)
    return result


def missing_coverage_keys(test_groups: list[dict[str, Any]], coverage_root: Path) -> list[str]:
    """Return expected test keys that have no collected coverage files."""
    available = _coverage_dirs(coverage_root)
    return sorted(expected_coverage_keys(test_groups) - available.keys())


def backfill_missing_coverage(
    test_groups: list[dict[str, Any]],
    current_root: Path,
    previous_root: Path,
) -> list[str]:
    """Copy only missing per-test coverage directories from a previous package."""
    missing = missing_coverage_keys(test_groups, current_root)
    previous = _coverage_dirs(previous_root)

    for key in missing:
        sources = previous.get(key)
        if not sources:
            continue
        destination = current_root / key / "covdata"
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(source)
                target = destination / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    stem = target.name
                    suffix = 1
                    while target.exists():
                        target = target.with_name(f"{stem}.historical-{suffix}")
                        suffix += 1
                shutil.copy2(path, target)

    return missing_coverage_keys(test_groups, current_root)


def _load_test_groups(raw_json: str) -> list[dict[str, Any]]:
    value = json.loads(raw_json)
    if not isinstance(value, list) or not all(isinstance(group, dict) for group in value):
        raise ValueError("test groups must be a JSON array of objects")
    return value


def _write_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


def _write_missing_file(path: Path, missing: list[str]) -> None:
    path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "backfill"))
    parser.add_argument("--test-groups-json", required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--missing-file", type=Path, default=Path("missing-coverage-tests.txt"))
    args = parser.parse_args()

    test_groups = _load_test_groups(args.test_groups_json)
    if args.command == "backfill":
        if args.previous_root is None:
            parser.error("--previous-root is required for backfill")
        missing = backfill_missing_coverage(test_groups, args.current_root, args.previous_root)
    else:
        missing = missing_coverage_keys(test_groups, args.current_root)

    _write_missing_file(args.missing_file, missing)
    _write_github_output("has_missing", str(bool(missing)).lower())
    _write_github_output("missing_count", str(len(missing)))

    if missing:
        print(f"Missing coverage data for {len(missing)} test target(s):")
        for key in missing:
            print(f"  - {key}")
        if args.command == "backfill":
            print(
                f"::warning::Coverage data is still missing for {len(missing)} "
                "test target(s) after OBS backfill; uploading the available data."
            )
    else:
        print("Coverage data is complete for all selected test targets.")


if __name__ == "__main__":
    main()
