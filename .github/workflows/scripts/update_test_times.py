#!/usr/bin/env python3
"""
Update test estimated times in config.yaml based on actual execution data from test_stats.json.

This script implements a Data-Driven Adaptive Scheduler that:
1. Detects anomalies and skips suspicious updates
2. Uses adaptive weighting based on deviation magnitude
3. Applies Exponential Moving Average (EMA) for smooth updates
"""

import argparse
import json
import sys
from pathlib import Path

from ruamel.yaml import YAML


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


# Anomaly detection thresholds
MAX_DURATION_THRESHOLD = 3600  # 1 hour - anything longer is suspicious
MIN_DURATION_THRESHOLD = 1  # Less than 1 second is suspicious for a test file


def load_test_stats(stats_path: Path) -> dict[str, dict]:
    """Load test stats from JSON file and return as dict keyed by test name."""
    if not stats_path.exists():
        print(f"{Colors.FAIL}Error: Stats file not found: {stats_path}{Colors.ENDC}")
        sys.exit(1)

    with open(stats_path) as f:
        stats_list = json.load(f)

    # Convert list to dict keyed by test name for easy lookup
    return {stat["name"]: stat for stat in stats_list}


def load_config(config_path: Path) -> tuple[dict, YAML]:
    """Load config.yaml preserving comments and structure."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=2)

    if not config_path.exists():
        print(f"{Colors.FAIL}Error: Config file not found: {config_path}{Colors.ENDC}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.load(f)

    return config, yaml


def save_config(config: dict, config_path: Path, yaml: YAML) -> None:
    """Save config back to YAML file preserving structure."""
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def is_anomaly(actual: float, old: float, test_name: str) -> bool:
    """
    Check if the new duration is anomalous.

    Returns True if anomaly detected (should skip update).
    """
    # Check absolute thresholds
    if actual > MAX_DURATION_THRESHOLD or actual < MIN_DURATION_THRESHOLD:
        print(
            f"⚠️ Anomaly detected for [{test_name}]: Old=[{old}]s, New=[{actual}]s. "
            "Skipping update (Manual check required)."
        )
        return True

    return False


def calculate_alpha(actual: float, old: float) -> float:
    """
    Calculate adaptive weight based on deviation magnitude.

    Returns alpha value for EMA calculation.
    """
    if old == 0:
        return 1.0  # New test, use actual value directly

    diff_ratio = abs(actual - old) / old

    if diff_ratio > 0.5:
        # Significant change (>50%): Trust new data more
        return 0.8
    else:
        # Minor fluctuation: Trust history more
        return 0.2


def update_test_times(config_path: Path, stats_path: Path, dry_run: bool = False) -> None:
    """
    Update estimated times in config based on actual execution stats.

    Args:
        config_path: Path to config.yaml
        stats_path: Path to test_stats.json
        dry_run: If True, only print what would be changed without saving
    """
    stats = load_test_stats(stats_path)
    config, yaml = load_config(config_path)

    updates_made = 0
    anomalies_found = 0
    new_tests_found = []
    tests_in_config = set()

    print(f"\n{Colors.HEADER}{'=' * 60}")
    print("Updating Test Estimated Times")
    print(f"{'=' * 60}{Colors.ENDC}\n")

    # Process each suite in config
    for suite_name, test_list in config.items():
        if not isinstance(test_list, list):
            continue

        for test_entry in test_list:
            if not isinstance(test_entry, dict) or "name" not in test_entry:
                continue

            test_name = test_entry["name"]
            tests_in_config.add(test_name)
            old_time = test_entry.get("estimated_time", 0)

            # Check if we have stats for this test
            if test_name not in stats:
                continue

            stat = stats[test_name]

            # Only update for passed tests
            if stat["status"] != "passed":
                print(
                    f"{Colors.OKCYAN}ℹ️ Skipping [{test_name}]: "
                    f"Test failed, keeping old estimate ({old_time}s){Colors.ENDC}"
                )
                continue

            actual_time = stat["duration"]

            # Rule A: Anomaly Detection
            if is_anomaly(actual_time, old_time, test_name):
                anomalies_found += 1
                continue

            # Handle new tests (estimated_time is 0 or missing)
            if old_time == 0:
                new_time = round(actual_time)
                print(
                    f"{Colors.OKGREEN}✨ New test [{test_name}]: "
                    f"Setting initial estimate to {new_time}s{Colors.ENDC}"
                )
            else:
                # Rule B & C: Adaptive Weighting + EMA
                alpha = calculate_alpha(actual_time, old_time)
                new_time_float = (actual_time * alpha) + (old_time * (1 - alpha))
                new_time = round(new_time_float)

                if new_time != old_time:
                    diff_ratio = abs(actual_time - old_time) / old_time * 100
                    print(
                        f"{Colors.OKBLUE}📊 [{test_name}]: "
                        f"Old={old_time}s → New={new_time}s "
                        f"(actual={actual_time:.1f}s, α={alpha}, diff={diff_ratio:.1f}%){Colors.ENDC}"
                    )

            if new_time != old_time:
                test_entry["estimated_time"] = new_time
                updates_made += 1

    # Check for new tests in stats that aren't in config
    for test_name in stats:
        if test_name not in tests_in_config:
            stat = stats[test_name]
            new_tests_found.append((test_name, stat["duration"], stat["status"]))

    # Print summary
    print(f"\n{Colors.HEADER}{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}{Colors.ENDC}")
    print(f"  Updates made: {updates_made}")
    print(f"  Anomalies skipped: {anomalies_found}")

    if new_tests_found:
        print(f"\n{Colors.WARNING}📋 New tests found in stats but not in config:{Colors.ENDC}")
        for name, duration, status in new_tests_found:
            print(f"  - {name} (duration={duration:.1f}s, status={status})")
        print(f"{Colors.WARNING}  Please add these to the appropriate suite in config.yaml{Colors.ENDC}")

    # Save updated config
    if not dry_run and updates_made > 0:
        save_config(config, config_path, yaml)
        print(f"\n{Colors.OKGREEN}✅ Config saved to {config_path}{Colors.ENDC}")
    elif dry_run:
        print(f"\n{Colors.WARNING}🔍 Dry run - no changes saved{Colors.ENDC}")
    else:
        print(f"\n{Colors.OKCYAN}ℹ️ No updates needed{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="Update test estimated times based on actual execution data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml (default: scripts/config.yaml)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path.cwd() / "test_stats.json",
        help="Path to test_stats.json (default: ./test_stats.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without saving",
    )
    args = parser.parse_args()

    update_test_times(args.config, args.stats, args.dry_run)


if __name__ == "__main__":
    main()
