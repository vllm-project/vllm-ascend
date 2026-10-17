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
"""Build dual-runner A2 test groups from a full-suite selection.

Temporary helper for the A2 dual-runner validation PR. It consumes the
``test_groups`` JSON produced by ``select_tests.py --all-tests``, keeps only
the groups whose ``npu_type`` is ``a2`` (the one_card suite; a3 / 310p / cpu
groups are dropped), and rewrites each surviving group for the two new runner
pools (``a2b1`` and ``a2b1b3``) so the identical A2 suite runs once per pool.

Outputs (GITHUB_OUTPUT when set, stdout otherwise):
  has_tests_a2        - "true"/"false"
  test_groups_a2b1    - JSON array of groups routed to the a2b1 pool
  test_groups_a2b1b3  - JSON array of groups routed to the a2b1b3 pool
  csrc_cache_target_ids - JSON array of csrc cache targets used by A2 groups
"""

from __future__ import annotations

import argparse
import json
import os
import sys

A2_NPU_TYPE = "a2"


def filter_a2_groups(groups: list[dict]) -> list[dict]:
    """Return only the groups whose npu_type is a2."""
    return [group for group in groups if group.get("npu_type") == A2_NPU_TYPE]


def rewrite_runner(groups: list[dict], runner_label: str, partition_suffix: str) -> list[dict]:
    """Point every group at *runner_label* and tag its partition name.

    Everything else (tests, buckets, num_npus, image_tag, csrc cache target)
    is preserved so both pools execute the exact same 7-bucket A2 suite.
    """
    rewritten = []
    for group in groups:
        new_group = dict(group)
        new_group["runner"] = runner_label
        new_group["partition_name"] = f"{group['partition_name']}-{partition_suffix}"
        rewritten.append(new_group)
    return rewritten


def collect_csrc_cache_targets(groups: list[dict]) -> list[str]:
    """Collect the deduplicated, sorted csrc cache targets of *groups*."""
    return sorted({group["csrc_cache_target"] for group in groups if group.get("csrc_cache_target")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--test-groups-json",
        required=True,
        help="test_groups JSON array emitted by select_tests.py",
    )
    parser.add_argument(
        "--a2b1-runner-label",
        required=True,
        help="Exact runner label for the a2b1 pool (from runner_label.json)",
    )
    parser.add_argument(
        "--a2b1b3-runner-label",
        required=True,
        help="Exact runner label for the a2b1b3 pool (from runner_label.json)",
    )
    args = parser.parse_args(argv)

    groups = json.loads(args.test_groups_json)
    a2_groups = filter_a2_groups(groups)
    a2b1_groups = rewrite_runner(a2_groups, args.a2b1_runner_label, "a2b1")
    a2b1b3_groups = rewrite_runner(a2_groups, args.a2b1b3_runner_label, "a2b1b3")

    outputs = {
        "has_tests_a2": str(bool(a2_groups)).lower(),
        "test_groups_a2b1": json.dumps(a2b1_groups, separators=(",", ":")),
        "test_groups_a2b1b3": json.dumps(a2b1b3_groups, separators=(",", ":")),
        "csrc_cache_target_ids": json.dumps(collect_csrc_cache_targets(a2_groups), separators=(",", ":")),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output_file:
            for key, value in outputs.items():
                print(f"{key}={value}", file=output_file)
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")

    print(f"A2 dual-runner groups: {len(a2_groups)} a2 groups of {len(groups)} total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
