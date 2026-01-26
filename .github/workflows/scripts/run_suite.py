import argparse
import glob
from pathlib import Path

import tabulate
from sglang.test.ci.ci_utils import TestFile, run_unittest_files

# NOTE: please sort the test cases alphabetically by the test file name
suites = {
    "e2e-singlecard": [
        TestFile("tests/e2e/singlecard/test_auto_fit_max_mode_len.py", 25),
        TestFile("tests/e2e/singlecard/test_aclgraph_accuracy.py", 480),
        TestFile("tests/e2e/singlecard/test_aclgraph_batch_invariant.py", 410),
        TestFile("tests/e2e/singlecard/test_aclgraph_mem.py", 130),
        TestFile("tests/e2e/singlecard/test_async_scheduling.py", 150),
        TestFile("tests/e2e/singlecard/test_batch_invariant.py", 320),
        TestFile("tests/e2e/singlecard/test_camem.py", 77),
        TestFile("tests/e2e/singlecard/test_completion_with_prompt_embeds.py", 76),
        TestFile("tests/e2e/singlecard/test_cpu_offloading.py", 132),
        TestFile("tests/e2e/singlecard/test_guided_decoding.py", 354),
        TestFile("tests/e2e/singlecard/test_ilama_lora.py", 95),
        TestFile("tests/e2e/singlecard/test_llama32_lora.py", 162),
        TestFile("tests/e2e/singlecard/test_qwen3_multi_loras.py", 600),
        TestFile("tests/e2e/singlecard/test_models.py", 600),
        TestFile("tests/e2e/singlecard/test_multistream_overlap_shared_expert.py", 600),
        TestFile("tests/e2e/singlecard/test_profile_execute_duration.py", 600),
        TestFile("tests/e2e/singlecard/test_quantization.py", 600),
        TestFile("tests/e2e/singlecard/test_sampler.py", 600),
        TestFile("tests/e2e/singlecard/test_vlm.py", 600),
        TestFile("tests/e2e/singlecard/test_xlite.py", 600),
        TestFile("tests/e2e/singlecard/compile/test_norm_quant_fusion.py", 600),
        TestFile("tests/e2e/singlecard/pooling/test_classification.py", 600),
        TestFile("tests/e2e/singlecard/pooling/test_embedding.py", 600),
        TestFile("tests/e2e/singlecard/pooling/test_scoring.py", 600),
        TestFile("tests/e2e/singlecard/spec_decode/test_mtp_eagle_correctness.py", 600),
        TestFile("tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py", 600),
    ],
}


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
    return [files[i] for i in indices]


def _sanity_check_suites(suites):
    dir_base = Path(__file__).parent
    project_root = dir_base.parent.parent
    disk_files = set([str(x.relative_to(dir_base)) for x in dir_base.glob("**/*.py") if x.name.startswith("test_")])

    suite_files = set([test_file.name for _, suite in suites.items() for test_file in suite])

    # Check for files in .github/scripts
    suite_files_local = {f for f in suite_files if not f.startswith("tests/")}
    missing_files = sorted(list(disk_files - suite_files_local))
    missing_text = "\n".join(f'TestFile("{x}"),' for x in missing_files)
    assert len(missing_files) == 0, (
        f"Some test files are not in test suite. "
        f"If this is intentional, please add the following to `not_in_ci` section:\n"
        f"{missing_text}"
    )

    # Check existence
    nonexistent_files = []
    for f in suite_files:
        if f.startswith("tests/"):
            if not (project_root / f).exists():
                nonexistent_files.append(f)
        else:
            if f not in disk_files:
                nonexistent_files.append(f)

    nonexistent_files.sort()
    nonexistent_text = "\n".join(f'TestFile("{x}"),' for x in nonexistent_files)
    assert len(nonexistent_files) == 0, f"Some test files in test suite do not exist on disk:\n{nonexistent_text}"

    not_in_ci_files = set([test_file.name for test_file in suites.get("__not_in_ci__", [])])
    in_ci_files = set(
        [test_file.name for suite_name, suite in suites.items() if suite_name != "__not_in_ci__" for test_file in suite]
    )
    intersection = not_in_ci_files & in_ci_files
    intersection_text = "\n".join(f'TestFile("{x}"),' for x in intersection)
    assert len(intersection) == 0, (
        f"Some test files are in both `not_in_ci` section and other suites:\n{intersection_text}"
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds.",
    )
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
        default=False,
        help="Continue running remaining tests even if one fails (useful for nightly tests)",
    )
    arg_parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    arg_parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    arg_parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    arg_parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600)",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    _sanity_check_suites(suites)

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

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

    print(msg, flush=True)

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    exit_code = run_unittest_files(
        files,
        timeout,
        args.continue_on_error,
        args.enable_retry,
        args.max_attempts,
        args.retry_wait_seconds,
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
