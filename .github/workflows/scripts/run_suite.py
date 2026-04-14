import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tabulate
import yaml
from ci_utils import TestFile, TestRecord, run_tests

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG_UPSTREAM = Path(__file__).parent / "upstream_config.yaml"

# Each entry: (config_path, path_prefix or None)
# path_prefix is prepended to every test name when loading that config.
_DEFAULT_CONFIGS: list[tuple[Path, str | None]] = [
    (_CONFIG_PATH, None),
    (_CONFIG_UPSTREAM, "vllm-empty"),
]


def load_suites(
    config_paths: list[tuple[Path, str | None]] | None = None,
) -> dict[str, list[TestFile]]:
    """Load all test suites from config yaml files.

    Each entry in config_paths is a (path, path_prefix) tuple.
    path_prefix, when set, is prepended to every test name in that config
    (e.g. "vllm-empty" turns "tests/foo.py" into "vllm-empty/tests/foo.py").
    """
    if config_paths is None:
        config_paths = _DEFAULT_CONFIGS

    all_suites: dict[str, list[TestFile]] = {}
    for config_path, path_prefix in config_paths:
        if not config_path.exists():
            continue
        print(f"Loading suites from {config_path}" + (f" (prefix: {path_prefix})" if path_prefix else ""))
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        for suite_name, tests in data.items():
            files = []
            for entry in tests or []:
                name = entry["name"]
                if path_prefix:
                    name = f"{path_prefix}/{name}"
                files.append(
                    TestFile(
                        name=name,
                        estimated_time=entry.get("estimated_time", 60),
                        is_skipped=entry.get("is_skipped", False),
                    )
                )
            all_suites[suite_name] = files
    return all_suites


def partition(files: list[TestFile], rank: int, size: int) -> list[TestFile]:
    """
    Split non-skipped files into `size` groups of approximately equal estimated
    time using a greedy algorithm, and return the group at index `rank`.
    Files within the returned group are sorted ascending by estimated_time.
    """
    active = [f for f in files if not f.is_skipped]
    if not active or size <= 0 or size > len(active):
        return []

    # Sort descending by weight; use original index as tiebreaker to be stable
    indexed = sorted(enumerate(active), key=lambda x: (-x[1].estimated_time, x[0]))

    buckets: list[list[int]] = [[] for _ in range(size)]
    sums = [0.0] * size

    for idx, test in indexed:
        lightest = sums.index(min(sums))
        buckets[lightest].append(idx)
        sums[lightest] += test.estimated_time

    return sorted([active[i] for i in buckets[rank]], key=lambda f: f.estimated_time)


def _find_project_root() -> Path:
    root = Path.cwd()
    if (root / "tests").exists():
        return root
    # Fall back: assume script lives at .github/workflows/scripts/
    return Path(__file__).parents[3]


def _minimal_covered_dirs(file_paths: set[str], root: Path) -> set[Path]:
    """Return the minimal set of directories that covers all file_paths."""
    dirs: set[Path] = set()
    for fp in file_paths:
        candidate = (root / fp).parent
        if not candidate.exists():
            continue
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            continue
        # Drop any existing entries that are subdirectories of rel
        dirs = {d for d in dirs if rel not in d.parents}
        # Only add rel if no ancestor already covers it
        if not any(d == rel or d in rel.parents for d in dirs):
            dirs.add(rel)
    return dirs


def sanity_check(suites: dict[str, list[TestFile]]) -> None:
    """
    Verify that:
    1. Every local test file in any suite exists on disk.
    2. No test_*.py files exist on disk (in covered dirs) that are absent from all suites.

    Files whose paths begin with an external prefix (e.g. "vllm-empty/") are skipped entirely —
    they live in a separate checkout and we only register a subset of them.
    Raises SystemExit with a descriptive message on failure.
    """
    external_prefixes = {prefix for _, prefix in _DEFAULT_CONFIGS if prefix}
    suite_files = {f.name.split("::")[0] for tests in suites.values() for f in tests}

    # Only check files that belong to this repo (no external prefix).
    local_files = {f for f in suite_files if not any(f.startswith(p + "/") for p in external_prefixes)}

    root = _find_project_root()
    covered = _minimal_covered_dirs(local_files, root)
    disk_files = {str(p.relative_to(root)) for d in covered for p in (root / d).rglob("test_*.py")}

    missing_from_suite = sorted(disk_files - local_files)
    if missing_from_suite:
        entries = "\n".join(f'  TestFile("{f}"),' for f in missing_from_suite)
        raise SystemExit(f"Test files on disk are not in any suite (add them or mark is_skipped=True):\n{entries}")

    missing_from_disk = sorted(local_files - disk_files)
    if missing_from_disk:
        entries = "\n".join(f'  TestFile("{f}"),' for f in missing_from_disk)
        raise SystemExit(f"Test files listed in suite do not exist on disk:\n{entries}")


def _print_plan(
    suite: str,
    files: list[TestFile],
    skipped: list[TestFile],
    partition_info: str,
) -> None:
    print(tabulate.tabulate([[suite, partition_info]], headers=["Suite", "Partition"], tablefmt="psql"))
    total_est = sum(f.estimated_time for f in files)
    print(f"✅ Enabled {len(files)} test(s)  (est. total {total_est:.1f}s):")
    for f in files:
        print(f"  - {f.name}  (est={f.estimated_time}s)")
    if skipped:
        print(f"\n❌ Skipped {len(skipped)} test(s) (consider recovering):")
        for f in skipped:
            print(f"  - {f.name}")
    print(flush=True)


def _print_results(
    suite: str,
    records: list[TestRecord],
    skipped: list[TestFile],
    partition_info: str,
) -> None:
    print(tabulate.tabulate([[suite, partition_info]], headers=["Suite", "Partition"], tablefmt="psql"))
    total_elapsed = sum(r.elapsed for r in records)
    passed_count = sum(1 for r in records if r.passed)
    print(f"Results: {passed_count}/{len(records)} passed  (actual total {total_elapsed:.1f}s):")
    for r in records:
        status = "✅ PASSED" if r.passed else "❌ FAILED"
        print(f"  {status}  {r.name}  (actual={r.elapsed:.0f}s  est={r.estimated:.0f}s)")
    if skipped:
        print(f"\n❌ Skipped {len(skipped)} test(s) (consider recovering):")
        for f in skipped:
            print(f"  - {f.name}")
    print(flush=True)


def _save_timing_json(
    records: list[TestRecord],
    suite: str,
    partition_id: int | None,
    partition_size: int | None,
    output_path: Path,
) -> None:
    passed_suites = [r.to_dict() for r in records if r.passed]
    payload = {
        "suite": suite,
        "partition_id": partition_id,
        "partition_size": partition_size,
        "commit_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": passed_suites,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Timing data written to {output_path}  ({len(passed_suites)}/{len(records)} passed)",
        flush=True,
    )


def main() -> None:
    suites = load_suites()

    parser = argparse.ArgumentParser(description="Run a named e2e test suite")
    parser.add_argument(
        "--suite",
        required=True,
        choices=list(suites.keys()),
        help="Name of the test suite to run",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        default=None,
        metavar="ID",
        help="Zero-based partition index (requires --auto-partition-size)",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        default=None,
        metavar="N",
        help="Total number of partitions",
    )
    parser.add_argument(
        "--auto-upgrade-estimated-times",
        action="store_true",
        help="Automatically update estimated times in config.yaml based on actual timings (default: False) \
If enabled, the script always exit with 0, even if some tests fail, since the primary purpose is to gather \
timing data to improve estimates.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running after a test failure (default: True)",
    )
    parser.add_argument(
        "--timing-report-json",
        type=Path,
        default=Path("test_timing_data.json"),
        help="Path to write the JSON timing data for CI aggregation",
    )
    args = parser.parse_args()

    sanity_check(suites)

    all_files = suites[args.suite]
    skipped = [f for f in all_files if f.is_skipped]

    if args.auto_partition_size is not None:
        files = partition(all_files, args.auto_partition_id, args.auto_partition_size)
        partition_info = f"{args.auto_partition_id + 1}/{args.auto_partition_size}"
    else:
        files = [f for f in all_files if not f.is_skipped]
        partition_info = "full"

    _print_plan(args.suite, files, skipped, partition_info)

    exit_code, records = run_tests(
        files,
        continue_on_error=args.continue_on_error,
    )

    _save_timing_json(records, args.suite, args.auto_partition_id, args.auto_partition_size, args.timing_report_json)

    _print_results(args.suite, records, skipped, partition_info)
    if args.auto_upgrade_estimated_times:
        sys.exit(0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
