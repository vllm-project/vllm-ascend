#!/usr/bin/env python3
"""Check and backfill per-test coverage data for full coverage runs.

A test target is considered "available" only when its ``covdata/`` directory
exists, contains files, and does NOT carry a ``FAILED`` sentinel. The runner
(``run_selected_tests.sh``) writes that sentinel when a target fails, so a
failed run's partial coverage is treated as unusable and replaced with the
OBS historical coverage during backfill (rather than shipped as-is).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

FAILED_SENTINEL = "FAILED"
EXPECTED_SENTINEL = "EXPECTED"


def coverage_key(target: str) -> str:
    """Return the output directory name used by run_selected_tests.sh."""
    basename = target[:-3] if target.endswith(".py") else target
    return basename.replace("\\", "/").replace("/", "__").replace("::", "--")


def expected_coverage_keys(
    test_groups: list[dict[str, Any]],
    coverage_root: Path | None = None,
) -> set[str]:
    """Collect the expected coverage keys.

    Case-level coverage runners write an ``EXPECTED`` sentinel before running
    each collected pytest nodeid. Prefer those keys when present. Falling back
    to test groups keeps existing file-level coverage packages compatible.
    """
    if coverage_root is not None and coverage_root.exists():
        case_keys = {
            path.parent.name
            for path in coverage_root.rglob(EXPECTED_SENTINEL)
            if path.is_file()
        }
        if case_keys:
            return case_keys

    keys: set[str] = set()
    for group in test_groups:
        if group.get("npu_type") == "cpu":
            if group.get("tests", "").split():
                keys.add("cpu-ut")
            continue
        keys.update(coverage_key(target) for target in group.get("tests", "").split())
    return keys


def _scan_coverage(root: Path) -> tuple[dict[str, list[Path]], set[str]]:
    """Return ``(dirs_by_key, failed_keys)`` found under *root*.

    ``dirs_by_key`` maps each coverage key to its ``covdata`` directories that
    contain at least one file. ``failed_keys`` holds keys whose ``covdata``
    directory contains a ``FAILED`` sentinel.
    """
    dirs_by_key: dict[str, list[Path]] = {}
    failed: set[str] = set()
    if not root.exists():
        return dirs_by_key, failed
    for covdata_dir in root.rglob("covdata"):
        if not covdata_dir.is_dir():
            continue
        files = [path for path in covdata_dir.rglob("*") if path.is_file()]
        if not files:
            continue
        key = covdata_dir.parent.name
        dirs_by_key.setdefault(key, []).append(covdata_dir)
        if any(path.name == FAILED_SENTINEL for path in files):
            failed.add(key)
    return dirs_by_key, failed


def missing_coverage_keys(test_groups: list[dict[str, Any]], coverage_root: Path) -> list[str]:
    """Return expected test keys that have no usable coverage files.

    A key is unusable when it has no ``covdata`` directory at all OR when its
    ``covdata`` directory carries a ``FAILED`` sentinel (the run failed, so its
    partial coverage must be replaced from history).
    """
    dirs_by_key, failed = _scan_coverage(coverage_root)
    usable = set(dirs_by_key) - failed
    return sorted(expected_coverage_keys(test_groups, coverage_root) - usable)


def backfill_missing_coverage(
    test_groups: list[dict[str, Any]],
    current_root: Path,
    previous_root: Path,
) -> list[str]:
    """Replace missing/failed per-test coverage from a previous package.

    For each missing key (absent or ``FAILED``), any existing ``covdata``
    directory under *current_root* is wiped first so the failed run's partial
    coverage (and its sentinel) is discarded, then the historical files are
    copied in from *previous_root*.
    """
    missing = missing_coverage_keys(test_groups, current_root)
    previous_dirs, _ = _scan_coverage(previous_root)

    for key in missing:
        sources = previous_dirs.get(key)
        if not sources:
            continue
        destination = current_root / key / "covdata"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative_path = path.relative_to(source)
                target = destination / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
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
