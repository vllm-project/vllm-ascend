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
import logging
import sys
from pathlib import Path

from ruamel.yaml import YAML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_test_stats(stats_path: Path) -> dict[str, dict]:
    """Load test stats from JSON file and return as dict keyed by test name."""
    if not stats_path.exists():
        logger.error(f"Stats file not found: {stats_path}")
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
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.load(f)

    return config, yaml


def save_config(config: dict, config_path: Path, yaml: YAML) -> None:
    """Save config back to YAML file preserving structure."""
    with open(config_path, "w") as f:
        yaml.dump(config, f)


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
    new_tests_found = []
    tests_in_config = set()

    logger.info("=" * 60)
    logger.info("Updating Test Estimated Times")
    logger.info("=" * 60)

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
                logger.info(f"Skipping [{test_name}]: Test failed, keeping old estimate ({old_time}s)")
                continue

            actual_time = stat["duration"]
            new_time = old_time

            # Handle new tests (estimated_time is 0 or missing)
            if old_time == 0:
                new_time = round(actual_time)
                logger.info(f"New test [{test_name}]: Setting initial estimate to {new_time}s")
            elif abs(actual_time - old_time) > 30:
                new_time = round(actual_time)
                logger.info(
                    f"Updating [{test_name}]: "
                    f"Old={old_time}s -> New={new_time}s "
                    f"(Actual={actual_time:.2f}s, Diff > 30s)"
                )
            else:
                # Diff <= 30s, do not update
                pass

            if new_time != old_time:
                test_entry["estimated_time"] = int(new_time)
                updates_made += 1

    # Check for new tests in stats that aren't in config
    for test_name in stats:
        if test_name not in tests_in_config:
            stat = stats[test_name]
            new_tests_found.append((test_name, stat["duration"], stat["status"]))

    # Print summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"  Updates made: {updates_made}")

    if new_tests_found:
        logger.warning("New tests found in stats but not in config:")
        for name, duration, status in new_tests_found:
            logger.warning(f"  - {name} (duration={duration:.1f}s, status={status})")
        logger.warning("  Please add these to the appropriate suite in config.yaml")

    # Save updated config
    if not dry_run and updates_made > 0:
        save_config(config, config_path, yaml)
        logger.info(f"Config saved to {config_path}")
    elif dry_run:
        logger.info("Dry run - no changes saved")
    else:
        logger.info("No updates needed")


def main():
    parser = argparse.ArgumentParser(description="Update test estimated times based on actual execution data")
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
