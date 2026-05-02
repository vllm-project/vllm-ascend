#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""
Like run_suite.py, but for running upstream vLLM tests on NPU.

We keep a separate whitelist (vllm_upstream_config.yaml) so upstream and
downstream tests don't step on each other. The main difference is we need
to chdir into the vLLM checkout before running pytest, and we skip the
sanity_check since the test files live in the vLLM repo, not ours.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tabulate
import yaml
from ci_utils import TestFile, TestRecord, run_tests

_CONFIG_PATH = Path(__file__).parent / "vllm_upstream_config.yaml"


def load_suites(config_path: Path = _CONFIG_PATH) -> dict[str, list[TestFile]]:
    """Load all upstream test suites from vllm_upstream_config.yaml."""
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
    """Split files into roughly equal time buckets, same logic as run_suite.py."""
    active = [f for f in files if not f.is_skipped]
    if not active or size <= 0 or size > len(active):
        return []

    indexed = sorted(
        enumerate(active), key=lambda x: (-x[1].estimated_time, x[0])
    )
    buckets: list[list[int]] = [[] for _ in range(size)]
    sums = [0.0] * size

    for idx, test in indexed:
        lightest = sums.index(min(sums))
        buckets[lightest].append(idx)
        sums[lightest] += test.estimated_time

    return sorted(
        [active[i] for i in buckets[rank]], key=lambda f: f.estimated_time
    )


def _print_plan(
    suite: str,
    files: list[TestFile],
    skipped: list[TestFile],
    partition_info: str,
) -> None:
    print(
        tabulate.tabulate(
            [[suite, partition_info]],
            headers=["Suite", "Partition"],
            tablefmt="psql",
        )
    )
    total_est = sum(f.estimated_time for f in files)
    print(f"Enabled {len(files)} upstream test(s)  (est. total {total_est:.1f}s):")
    for f in files:
        print(f"  - {f.name}  (est={f.estimated_time}s)")
    if skipped:
        print(f"\nSkipped {len(skipped)} test(s):")
        for f in skipped:
            print(f"  - {f.name}")
    print(flush=True)


def _print_results(
    suite: str,
    records: list[TestRecord],
    skipped: list[TestFile],
    partition_info: str,
) -> None:
    print(
        tabulate.tabulate(
            [[suite, partition_info]],
            headers=["Suite", "Partition"],
            tablefmt="psql",
        )
    )
    total_elapsed = sum(r.elapsed for r in records)
    passed_count = sum(1 for r in records if r.passed)
    print(f"Results: {passed_count}/{len(records)} passed  (actual total {total_elapsed:.1f}s):")
    for r in records:
        status = "PASSED" if r.passed else "FAILED"
        print(f"  {status}  {r.name}  (actual={r.elapsed:.0f}s  est={r.estimated:.0f}s)")
    if skipped:
        print(f"\nSkipped {len(skipped)} test(s):")
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
        f"Timing data written to {output_path}  "
        f"({len(passed_suites)}/{len(records)} passed)",
        flush=True,
    )


def main() -> None:
    suites = load_suites()

    parser = argparse.ArgumentParser(
        description="Run upstream vLLM community tests on NPU via whitelist"
    )
    parser.add_argument(
        "--suite",
        required=True,
        choices=list(suites.keys()),
        help="Name of the upstream test suite to run",
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        required=True,
        help="Path to the vLLM checkout directory (tests are run from here)",
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
        "--continue-on-error",
        action="store_true",
        help="Continue running after a test failure",
    )
    parser.add_argument(
        "--timing-report-json",
        type=Path,
        default=Path("upstream_test_timing_data.json"),
        help="Path to write the JSON timing data for CI aggregation",
    )
    args = parser.parse_args()

    # Make sure the vLLM checkout actually has a tests/ dir
    vllm_root = args.vllm_root.resolve()
    if not (vllm_root / "tests").is_dir():
        raise SystemExit(
            f"vLLM root does not contain a tests/ directory: {vllm_root}"
        )

    # We need to be inside vLLM root for pytest to find the test files
    original_cwd = Path.cwd()
    os.chdir(vllm_root)
    print(f"Changed working directory to: {vllm_root}", flush=True)

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

    # Switch back before writing timing data
    os.chdir(original_cwd)

    _save_timing_json(
        records,
        args.suite,
        args.auto_partition_id,
        args.auto_partition_size,
        args.timing_report_json,
    )

    _print_results(args.suite, records, skipped, partition_info)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
