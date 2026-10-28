#!/usr/bin/env python3
"""
Update estimated_times.yaml from CI timing data.

Usage:
    python3 update_estimated_times.py \
        --timing-dir ./timing-artifacts \
        --config .github/workflows/scripts/estimated_times.yaml

Methodology:
  1. Collect all elapsed times per test from timing JSON files
  2. Take median per test
  3. Apply 10 % safety buffer, round to nearest 10 s
  4. Overwrite the estimated_times mapping in estimated_times.yaml
"""

import argparse
import json
from pathlib import Path

import yaml


def collect_timings(timing_dir: Path) -> dict[str, list[int]]:
    """Scan *timing_dir* recursively for timing JSON files.

    Returns ``{test_name: [elapsed_seconds, ...]}`` for all passed tests.
    """
    json_files = list(timing_dir.rglob("*.json"))
    print(f"Found {len(json_files)} timing file(s) in {timing_dir}")

    timings: dict[str, list[int]] = {}
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping {path}: {e}")
            continue

        if isinstance(data, dict):
            tests = data.get("tests", [])
        elif isinstance(data, list):
            tests = data
        else:
            continue

        for test in tests:
            name: str = test.get("name", "")
            passed: bool = test.get("passed", False)
            elapsed: float = test.get("elapsed", 0.0)
            if not name or not passed or elapsed <= 0:
                continue
            timings.setdefault(name, []).append(int(elapsed))

    return timings


def compute_median(values: list[int]) -> int:
    """Compute the median of a list of integers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) // 2
    return sorted_vals[n // 2]


def update_config(config_path: Path, timings: dict[str, list[int]]) -> int:
    """Overwrite the ``estimated_times`` mapping in *config_path*.

    For each test: median -> x1.1 -> round to nearest 10 s.
    Comment lines above ``estimated_times:`` are preserved.

    Returns the number of entries whose values changed.
    """
    text = config_path.read_text(encoding="utf-8")

    meta = yaml.safe_load(text) or {}
    existing: dict[str, int] = meta.get("estimated_times", {}) or {}

    # --- compute new entries (file-level only; drop legacy ``::nodeid`` entries) ---
    new_entries = {k: v for k, v in existing.items() if "::" not in k}
    changed = 0
    for name in sorted(timings.keys()):
        elapsed_list = timings[name]
        if not elapsed_list:
            continue
        median = compute_median(elapsed_list)
        new_val = int(round(median * 1.1 / 10.0) * 10.0)
        if new_val <= 0:
            new_val = 10
        # File-level granularity: bucketing and precision testing both work
        # on files, so method-level (``::nodeid``) estimates are not written.
        key = name.split("::", 1)[0]
        # Skip non-test entries (e.g. ``cpu-ut (115 targets)`` batch label)
        if not key.startswith("tests/"):
            continue
        if new_entries.get(key) != new_val:
            new_entries[key] = new_val
            changed += 1

    if not changed:
        print("No estimated_time values changed.")
        return 0

    lines = text.split("\n")
    et_start = None
    for i, line in enumerate(lines):
        if line.strip() == "estimated_times:":
            et_start = i
            break

    if et_start is None:
        print("Error: 'estimated_times:' section not found in config file.")
        return 0

    new_section_lines = ["estimated_times:"]
    for name, val in new_entries.items():
        new_section_lines.append(f"  {name}: {val}")

    new_text = "\n".join(lines[:et_start] + new_section_lines) + "\n"
    config_path.write_text(new_text, encoding="utf-8")
    print(f"\nDone. {changed} estimated_time value(s) changed.")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update estimated_times.yaml from CI timing data",
    )
    parser.add_argument(
        "--timing-dir",
        required=True,
        type=Path,
        help="Directory containing timing JSON files (searched recursively)",
    )
    parser.add_argument(
        "--config",
        default=".github/workflows/scripts/estimated_times.yaml",
        type=Path,
        help="Path to estimated_times.yaml",
    )
    args = parser.parse_args()

    timings = collect_timings(args.timing_dir)
    if not timings:
        print("No timing data collected. Exiting without changes.")
        return

    print(f"\nCollected timing data for {len(timings)} test(s).")
    print(f"Updating {args.config}...")
    update_config(args.config, timings)


if __name__ == "__main__":
    main()
