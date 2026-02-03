import argparse
import os
from pathlib import Path

import tabulate
import yaml
from ci_utils import TestFile, run_e2e_files


def load_suites_from_config(config_path: str = "config.yaml") -> dict[str, list[TestFile]]:
    # Get absolute path relative to this script
    script_dir = Path(__file__).parent
    abs_config_path = script_dir / config_path

    with open(abs_config_path) as f:
        suites_data = yaml.safe_load(f)

    suites = {}

    for suite_name, test_files in suites_data.items():
        suites[suite_name] = []
        for file_data in test_files:
            name = file_data.get("name")
            estimated_time = file_data.get("estimated_time", 60)
            is_skipped = file_data.get("is_skipped", False)
            suites[suite_name].append(TestFile(name, estimated_time, is_skipped))

    return suites


suites = load_suites_from_config()


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    # Filter out skipped files
    files = [f for f in files if not f.is_skipped]
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    indices.sort(key=lambda i: files[i].estimated_time)
    return [files[i] for i in indices]


def _get_disk_covered_dirs(all_suite_files: set[str], project_root: Path | str) -> list[str]:
    covered_dirs = set()
    for file_path in all_suite_files:
        # e.g. tests/e2e/singlecard/test_foo.py -> tests/e2e/singlecard
        parent_dir = (project_root / file_path).parent if os.path.isfile(file_path) else (project_root / file_path)
        if parent_dir.exists():
            # Store relative path to project root
            try:
                rel_dir = parent_dir.relative_to(project_root)

                # Check if this directory is already covered by a parent directory
                is_covered = False
                for existing_dir in list(covered_dirs):
                    # If existing_dir is a parent of rel_dir, rel_dir is already covered
                    if existing_dir in rel_dir.parents or existing_dir == rel_dir:
                        is_covered = True
                        break
                    # If rel_dir is a parent of existing_dir, replace existing_dir with rel_dir
                    elif rel_dir in existing_dir.parents:
                        covered_dirs.remove(existing_dir)
                        # We continue checking other existing_dirs, but we know rel_dir should be added
                        # unless another parent covers it (which is handled by the first if block logic effectively
                        # but we need to be careful with modification during iteration, so we use list copy)

                if not is_covered:
                    covered_dirs.add(rel_dir)

            except ValueError:
                pass
    return covered_dirs


def _sanity_check_suites(suites: dict[str, list[TestFile]]):
    """
    Check if all test files defined in the suites exist on disk.
    """
    # 1. Collect all test files defined in all suites
    all_suite_files = set()
    for suite in suites.values():
        for test_file in suite:
            # Handle ::test_case syntax
            file_path = test_file.name.split("::")[0]
            all_suite_files.add(file_path)

    # 2. Identify all directories covered by the suites
    project_root = Path.cwd()
    if not (project_root / "tests").exists():
        script_dir = Path(__file__).parent
        # .github/workflows/scripts -> ../../../ -> root
        project_root = script_dir.parents[2]
    # For now, we only check dirs under [tests/e2e/singlecard, tests/e2e/multicard]
    covered_dirs = _get_disk_covered_dirs(all_suite_files, project_root)

    # 3. Scan disk for all test_*.py files in these directories
    all_disk_files = set()
    for dir_path in covered_dirs:
        full_dir_path = project_root / dir_path
        # rglob is equivalent to glob('**/' + pattern)
        for py_file in full_dir_path.rglob("test_*.py"):
            try:
                rel_path = py_file.relative_to(project_root)
                all_disk_files.add(str(rel_path))
            except ValueError:
                pass

    # 4. Find files on disk but missing from ANY suite
    # We check if a disk file is present in 'all_suite_files' (union of all suites)
    missing_files = sorted(list(all_disk_files - all_suite_files))

    missing_text = "\n".join(f'TestFile("{x}"),' for x in missing_files)

    if missing_files:
        assert len(missing_files) == 0, (
            f"Some test files found on disk in covered directories are not in ANY test suite.\n"
            f"Scanned directories: {sorted([str(d) for d in covered_dirs])}\n"
            f"Missing files:\n"
            f"{missing_text}\n"
            f"If this is intentional, please label them as 'is_skipped=True' and add them to the test suite."
        )

    # 5. check if all files in suites exist on disk
    non_existent_files = sorted(list(all_suite_files - all_disk_files))
    non_existent_text = "\n".join(f'TestFile("{x}"),' for x in non_existent_files)
    assert len(non_existent_files) == 0, (
        f"Some test files in test suite do not exist on disk:\n"
        f"{non_existent_text}\n"
        f"Please check if the test files are correctly specified in the local repository."
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    arg_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue running remaining tests even if one fails (useful for nightly tests)",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    _sanity_check_suites(suites)
    files = suites[args.suite]

    files_disabled = [f for f in files if f.is_skipped]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)

    # Print test info at beginning (similar to test/run_suite.py pretty_print_tests)
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} (0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    headers = ["Suite", "Partition"]
    rows = [[args.suite, partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"

    total_est_time = sum(f.estimated_time for f in files)
    msg += f"✅ Enabled {len(files)} test(s) (est total {total_est_time:.1f}s):\n"
    for f in files:
        msg += f"  - {f.name} (est_time={f.estimated_time})\n"
    msg += f"\n❌ Disabled {len(files_disabled)} test(s)(Please consider to recover them):\n"
    for f in files_disabled:
        msg += f"  - {f.name} (est_time={f.estimated_time})\n"

    print(msg, flush=True)

    exit_code = run_e2e_files(
        files,
        continue_on_error=args.continue_on_error,
    )

    # Print tests again at the end for visibility
    msg = "\n" + tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
    msg += f"✅ Executed {len(files)} test(s) (est total {total_est_time:.1f}s):\n"
    for f in files:
        msg += f"  - {f.name} (est_time={f.estimated_time})\n"
    print(msg, flush=True)

    exit(exit_code)


if __name__ == "__main__":
    main()
