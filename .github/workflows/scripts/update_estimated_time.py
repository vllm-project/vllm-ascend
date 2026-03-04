#!/usr/bin/env python3
"""
Update estimated_time in config.yaml from CI timing data.

Usage:
    python3 update_estimated_time.py \
        --timing-dir ./timing-artifacts \
        --config .github/workflows/scripts/config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml


def collect_timings(timing_dir: Path) -> dict[str, int]:
    """
    Recursively scan timing_dir for JSON files produced by run_suite.py.
    Returns {test_name: elapsed_seconds} for all passed tests.
    Warns if the same test name appears in multiple files.
    """
    json_files = list(timing_dir.rglob("*.json"))
    print(f"Found {len(json_files)} timing file(s) in {timing_dir}")

    timings: dict[str, int] = {}
    for path in json_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping {path}: {e}")
            continue

        for test in data.get("tests", []):
            if not test.get("passed", False):
                continue
            name: str = test.get("name", "")
            elapsed: float = test.get("elapsed", 0.0)
            if not name or elapsed <= 0:
                continue
            if name in timings:
                print(f"  Warning: duplicate entry for '{name}', overwriting {timings[name]}s with {int(elapsed)}s")
            timings[name] = int(elapsed)

    return timings


def update_config(config_path: Path, timings: dict[str, int]) -> int:
    """
    Load config.yaml, update estimated_time for each test found in timings,
    and write the result back. Returns the number of changed entries.
    """
    configs: dict = yaml.safe_load(config_path.read_text())

    changed = 0
    for suite_tests in configs.values():
        for test in suite_tests:
            name: str = test.get("name", "")
            if name not in timings:
                continue
            old_time: int = test.get("estimated_time", 0)
            new_time: int = timings[name]
            if old_time == new_time:
                continue
            test["estimated_time"] = new_time
            print(f"  {name}: {old_time}s -> {new_time}s")
            changed += 1

    config_path.write_text(yaml.dump(configs, default_flow_style=False, allow_unicode=True, sort_keys=False))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Update estimated_time in config.yaml from CI timing data")
    parser.add_argument(
        "--timing-dir",
        required=True,
        type=Path,
        help="Directory containing timing JSON files (searched recursively)",
    )
    parser.add_argument(
        "--config",
        default=".github/workflows/scripts/config.yaml",
        type=Path,
        help="Path to config.yaml (default: .github/workflows/scripts/config.yaml)",
    )
    args = parser.parse_args()

    timings = collect_timings(args.timing_dir)
    if not timings:
        print("No timing data collected. Exiting without changes.")
        return

    print(f"\nCollected timing data for {len(timings)} test(s).")
    print(f"Updating {args.config}...")
    changed = update_config(args.config, timings)
    print(f"\nDone. {changed} estimated_time value(s) changed.")


if __name__ == "__main__":
    main()
