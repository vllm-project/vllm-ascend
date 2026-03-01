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


def load_suites(config_path: Path = _CONFIG_PATH) -> dict[str, list[TestFile]]:
    """Load all test suites from config.yaml."""
    data = yaml.safe_load(config_path.read_text())
    return {
        suite_name: [
            TestFile(
                name=entry["name"],
                estimated_time=entry.get("estimated_time", 60),
                is_skipped=entry.get("is_skipped", False),
            )
            for entry in entries
        ]
        for suite_name, entries in data.items()
    }


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
    1. Every test file in any suite exists on disk.
    2. No test_*.py files exist on disk (in covered dirs) that are absent from all suites.
    Raises SystemExit with a descriptive message on failure.
    """
    suite_files = {f.name.split("::")[0] for tests in suites.values() for f in tests}
    root = _find_project_root()
    covered = _minimal_covered_dirs(suite_files, root)

    disk_files = {str(p.relative_to(root)) for d in covered for p in (root / d).rglob("test_*.py")}

    missing_from_suite = sorted(disk_files - suite_files)
    if missing_from_suite:
        entries = "\n".join(f'  TestFile("{f}"),' for f in missing_from_suite)
        raise SystemExit(f"Test files on disk are not in any suite (add them or mark is_skipped=True):\n{entries}")

    missing_from_disk = sorted(suite_files - disk_files)
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
    passed = [r.to_dict() for r in records if r.passed]
    payload = {
        "suite": suite,
        "partition_id": partition_id,
        "partition_size": partition_size,
        "commit_sha": os.environ.get("GITHUB_SHA", ""),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": passed,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(
        f"Timing data written to {output_path}  ({len(passed)}/{len(records)} passed)",
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
        "--timing-report",
        type=Path,
        default=Path("test_timing_report.md"),
        help="Path to write the Markdown timing report",
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
        report_path=args.timing_report,
    )

    _save_timing_json(records, args.suite, args.auto_partition_id, args.auto_partition_size, args.timing_report_json)

    _print_results(args.suite, records, skipped, partition_info)
    if args.auto_upgrade_estimated_times:
        sys.exit(0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
