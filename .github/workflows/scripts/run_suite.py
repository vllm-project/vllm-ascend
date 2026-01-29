import argparse
import os
from pathlib import Path

import tabulate
from ci_utils import TestFile, run_e2e_files

# NOTE: Please add the case with the following format and give an expected time for each case:
# case_path, estimated_time, is_skipped, num_devices
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
        TestFile("tests/e2e/singlecard/test_qwen3_multi_loras.py", 65),
        TestFile("tests/e2e/singlecard/test_models.py", 300),
        TestFile("tests/e2e/singlecard/test_multistream_overlap_shared_expert.py", 200),
        TestFile("tests/e2e/singlecard/test_profile_execute_duration.py", 10),
        TestFile("tests/e2e/singlecard/test_quantization.py", 200),
        TestFile("tests/e2e/singlecard/test_sampler.py", 200),
        TestFile("tests/e2e/singlecard/test_vlm.py", 354),
        TestFile("tests/e2e/singlecard/test_xlite.py", 45),
        TestFile("tests/e2e/singlecard/compile/test_norm_quant_fusion.py", 70),
        TestFile("tests/e2e/singlecard/pooling/test_classification.py", 120),
        TestFile("tests/e2e/singlecard/pooling/test_embedding.py", 270),
        TestFile("tests/e2e/singlecard/pooling/test_scoring.py", 500),
        TestFile("tests/e2e/singlecard/spec_decode/test_mtp_eagle_correctness.py", 1500),
        TestFile("tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py", 1800),
        TestFile("tests/e2e/singlecard/model_runner_v2/test_basic.py", 80, is_skipped=True),
    ],
}

suites_singlecard_light = {
    "e2e-singlecard-light": [
        TestFile("tests/e2e/singlecard/test_aclgraph_accuracy.py::test_piecewise_res_consistency", 220),
        TestFile("tests/e2e/singlecard/test_quantization.py::test_qwen3_w8a8_quant", 90),
    ],
}

suites_2_card_light = {
    "e2e-2card-light": [
        TestFile("tests/e2e/multicard/2-cards/test_qwen3_moe.py::test_qwen3_moe_distributed_mp_tp2_ep", 220),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_deepseek3_2_w8a8_pruning_mtp_tp2_ep",
            90,
        ),
    ],
}

suites_2_card = {
    "e2e-multicard-2-cards": [
        # TODO: recover skipped tests
        TestFile("tests/e2e/multicard/2-cards/test_aclgraph_capture_replay.py", 0, is_skipped=True),
        TestFile("tests/e2e/multicard/2-cards/spec_decode/test_spec_decode.py", 0, is_skipped=True),
        TestFile("tests/e2e/multicard/2-cards/test_offline_weight_load.py", 0, is_skipped=True),
        TestFile("tests/e2e/multicard/2-cards/test_shared_expert_dp.py", 0, is_skipped=True),
        TestFile("tests/e2e/multicard/2-cards/test_qwen3_performance.py", 180),
        TestFile("tests/e2e/multicard/2-cards/test_data_parallel.py", 380),
        TestFile("tests/e2e/multicard/2-cards/test_expert_parallel.py", 170),
        TestFile("tests/e2e/multicard/2-cards/test_external_launcher.py", 300),
        TestFile("tests/e2e/multicard/2-cards/test_full_graph_mode.py", 400),
        TestFile("tests/e2e/multicard/2-cards/test_ilama_lora_tp2.py", 60),
        # Run the test in a separate step to avoid oom
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_deepseek_multistream_moe_tp2", 100
        ),
        TestFile("tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_w4a8_dynamic_tp2", 80),
        TestFile("tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_moe_sp_tp2", 132),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_deepseek_w4a8_accuracy_tp2", 132
        ),
        TestFile("tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_moe_fc2_tp2", 140),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_deepseek_v2_lite_fc1_tp2", 82
        ),
        TestFile("tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_dense_fc1_tp2", 73),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_dense_prefetch_mlp_weight_tp2",
            71,
        ),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_deepseek3_2_w8a8_pruning_mtp_tp2_ep",
            111,
        ),
        TestFile(
            "tests/e2e/multicard/2-cards/test_offline_inference_distributed.py::test_qwen3_w4a4_distributed_tp2", 180
        ),
        TestFile("tests/e2e/multicard/2-cards/test_pipeline_parallel.py", 270),
        TestFile("tests/e2e/multicard/2-cards/test_prefix_caching.py", 430),
        TestFile("tests/e2e/multicard/2-cards/test_quantization.py", 70),
        TestFile("tests/e2e/multicard/2-cards/test_qwen3_moe.py", 1050),
        TestFile("tests/e2e/multicard/2-cards/test_single_request_aclgraph.py", 215),
    ],
}

# TODO: recover skipped tests
suites_4_card = {
    "e2e-multicard-4-cards": [
        TestFile("tests/e2e/multicard/4-cards/test_qwen3_next.py", 1250),
        TestFile("tests/e2e/multicard/4-cards/test_data_parallel_tp2.py", 60, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/test_kimi_k2.py", 100, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/long_sequence/test_accuracy.py", 60, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/long_sequence/test_basic.py", 60, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/long_sequence/test_chunked_prefill.py", 60, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/long_sequence/test_mtp.py", 60, is_skipped=True),
        TestFile("tests/e2e/multicard/4-cards/spec_decode/test_mtp_qwen3_next.py", 60, is_skipped=True),
    ],
}

# TODO: add more suites
suites.update(suites_singlecard_light)
suites.update(suites_2_card_light)
suites.update(suites_2_card)
suites.update(suites_4_card)


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
        "--timeout-per-file",
        type=int,
        default=2000,
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

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    exit_code = run_e2e_files(
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
