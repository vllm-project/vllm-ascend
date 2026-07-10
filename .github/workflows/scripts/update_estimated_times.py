#!/usr/bin/env python3
"""
Update estimated_times in test_config.yaml from CI timing data.

Usage:
    python3 update_estimated_times.py \
        --timing-dir ./timing-artifacts \
        --config .github/workflows/scripts/test_config.yaml

Methodology:
  1. Collect all elapsed times per test from timing JSON files
  2. Take median per test
  3. Apply 10 % safety buffer, round to nearest 10 s
  4. Overwrite estimated_times section in test_config.yaml
"""

import argparse
import json
from pathlib import Path


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
    """Overwrite the ``estimated_times`` section in *config_path*.

    For each test: median -> x1.1 -> round to nearest 10 s.

    Returns the number of entries whose values changed.
    """
    text = config_path.read_text()

    # --- parse existing estimated_times ---
    import yaml

    docs = list(yaml.safe_load_all(text))
    existing: dict[str, int] = {}
    if len(docs) >= 2 and isinstance(docs[1], dict):
        existing = docs[1].get("estimated_times", {}) or {}

    # --- compute new entries (preserve existing, update from timing data) ---
    new_entries = dict(existing)
    changed = 0
    for name in sorted(timings.keys()):
        elapsed_list = timings[name]
        if not elapsed_list:
            continue
        median = compute_median(elapsed_list)
        new_val = int(round(median * 1.1 / 10.0) * 10.0)
        if new_val <= 0:
            new_val = 10
        # Preserve existing ``::nodeid`` entries if configured, else fall back to file-level
        if "::" in name and name in existing:
            key = name
        else:
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

    # --- find section boundaries in raw text ---
    lines = text.split("\n")
    et_start = None
    section_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "estimated_times:":
            et_start = i
        elif et_start is not None and section_end is None:
            # Next top-level key (no leading spaces) after estimated_times marks the end
            if line and not line.startswith(" ") and not line.startswith("#") and not line.startswith("-"):
                if ":" in line and line.split(":")[0].strip():
                    section_end = i
                    # Backtrack through preceding blank/comment lines so they
                    # are preserved in ``after`` rather than being dropped.
                    while section_end > 0 and (
                        lines[section_end - 1].strip() == "" or lines[section_end - 1].strip().startswith("#")
                    ):
                        section_end -= 1
                    break

    if et_start is None:
        print("Error: 'estimated_times:' section not found in config file.")
        return 0

    # Build new estimated_times lines
    new_section_lines = ["estimated_times:"]
    for name, val in new_entries.items():
        new_section_lines.append(f"  {name}: {val}")

    # Reconstruct file
    before = lines[:et_start]
    after = lines[section_end:] if section_end is not None else []

    new_text = "\n".join(before) + "\n" + "\n".join(new_section_lines) + "\n"
    if after:
        # Ensure a blank line separates estimated_times from the next section
        if after[0].strip():
            new_text += "\n"
        new_text += "\n".join(after) + "\n"

    config_path.write_text(new_text)
    print(f"\nDone. {changed} estimated_time value(s) changed.")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update estimated_times in test_config.yaml from CI timing data",
    )
    parser.add_argument(
        "--timing-dir",
        required=True,
        type=Path,
        help="Directory containing timing JSON files (searched recursively)",
    )
    parser.add_argument(
        "--config",
        default=".github/workflows/scripts/test_config.yaml",
        type=Path,
        help="Path to test_config.yaml",
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
