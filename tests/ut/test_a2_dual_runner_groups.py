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
"""Tests for the temporary A2 dual-runner selection flow.

Covers ``.github/workflows/scripts/a2_dual_runner_groups.py`` (pure
filter/rewrite logic) and its integration with the real full-suite output of
``select_tests.py --all-tests --skip-cpu-ut``: only the A2 (one_card) groups
survive, they are duplicated for the two new runner pools, and the 7-bucket
load balancing is preserved.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / ".github" / "workflows" / "scripts" / "a2_dual_runner_groups.py"
_SELECT_TESTS_PATH = _REPO_ROOT / ".github" / "workflows" / "scripts" / "select_tests.py"

A2B1_LABEL = "linux-aarch64-a2b1-1-hk-001"
A2B4_LABEL = "linux-aarch64-a2b4-1-hk-001"
BUCKET_COUNT = 7


def _load_module():
    spec = importlib.util.spec_from_file_location("a2_dual_runner_groups", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dual():
    return _load_module()


def _make_group(npu_type, runner, partition="1-1", tests=("a.py", "b.py"), **extra):
    group = {
        "num_npus": 1,
        "npu_type": npu_type,
        "runner": runner,
        "tests": " ".join(tests),
        "partition_name": f"{npu_type}-suite",
        "partition": partition,
    }
    group.update(extra)
    return group


class TestFilterA2Groups:
    def test_keeps_only_a2_groups(self, dual):
        groups = [
            _make_group("a2", "linux-aarch64-a2b3-1"),
            _make_group("a3", "linux-aarch64-a3-2"),
            _make_group("310p", "linux-aarch64-310p-1"),
            _make_group("cpu", "linux-amd64-cpu-8-hk"),
        ]
        filtered = dual.filter_a2_groups(groups)
        assert len(filtered) == 1
        assert filtered[0]["npu_type"] == "a2"

    def test_empty_and_missing_npu_type(self, dual):
        assert dual.filter_a2_groups([]) == []
        assert dual.filter_a2_groups([{"runner": "x"}]) == []


class TestRewriteRunner:
    def test_rewrites_runner_and_partition_name_only(self, dual):
        groups = [
            _make_group(
                "a2",
                "linux-aarch64-a2b3-1",
                partition="3-7",
                tests=("one.py", "two.py"),
                image_tag="9.1.0-910b-ubuntu22.04-py3.12",
                csrc_cache_target="a2-arm64-ubuntu",
            )
        ]
        rewritten = dual.rewrite_runner(groups, A2B1_LABEL, "a2b1")
        assert len(rewritten) == 1
        original, new = groups[0], rewritten[0]
        assert new["runner"] == A2B1_LABEL
        assert new["partition_name"] == original["partition_name"] + "-a2b1"
        # Everything that defines the execution is preserved.
        for key in ("tests", "partition", "num_npus", "npu_type", "image_tag", "csrc_cache_target"):
            assert new[key] == original[key]
        # The input group must not be mutated.
        assert original["runner"] == "linux-aarch64-a2b3-1"
        assert original["partition_name"] == "a2-suite"

    def test_preserves_bucket_order_and_count(self, dual):
        groups = [
            _make_group("a2", "linux-aarch64-a2b3-1", partition=f"{i}-{BUCKET_COUNT}")
            for i in range(1, BUCKET_COUNT + 1)
        ]
        rewritten = dual.rewrite_runner(groups, A2B4_LABEL, "a2b4")
        assert len(rewritten) == BUCKET_COUNT
        for original, new in zip(groups, rewritten):
            assert new["partition"] == original["partition"]
            assert new["tests"] == original["tests"]


class TestCollectCsrcCacheTargets:
    def test_dedup_and_sort(self, dual):
        groups = [
            _make_group("a2", "r", csrc_cache_target="a2-arm64-ubuntu"),
            _make_group("a2", "r", csrc_cache_target="a2-arm64-ubuntu"),
            _make_group("a2", "r"),  # no target
        ]
        assert dual.collect_csrc_cache_targets(groups) == ["a2-arm64-ubuntu"]

    def test_empty(self, dual):
        assert dual.collect_csrc_cache_targets([]) == []


class TestMain:
    def test_outputs_written_to_github_output(self, dual, tmp_path, monkeypatch):
        github_output = tmp_path / "github_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
        groups = [
            _make_group("a2", "linux-aarch64-a2b3-1", csrc_cache_target="a2-arm64-ubuntu"),
            _make_group("a3", "linux-aarch64-a3-2", csrc_cache_target="a3-arm64-ubuntu"),
        ]
        exit_code = dual.main(
            [
                "--test-groups-json",
                json.dumps(groups),
                "--a2b1-runner-label",
                A2B1_LABEL,
                "--a2b4-runner-label",
                A2B4_LABEL,
            ]
        )
        assert exit_code == 0

        outputs = {}
        for line in github_output.read_text().splitlines():
            key, _, value = line.partition("=")
            outputs[key] = value

        assert outputs["has_tests_a2"] == "true"
        b1_groups = json.loads(outputs["test_groups_a2b1"])
        b4_groups = json.loads(outputs["test_groups_a2b4"])
        # Only the a2 group survived, once per pool.
        assert len(b1_groups) == len(b4_groups) == 1
        assert b1_groups[0]["runner"] == A2B1_LABEL
        assert b4_groups[0]["runner"] == A2B4_LABEL
        assert b1_groups[0]["partition_name"].endswith("-a2b1")
        assert b4_groups[0]["partition_name"].endswith("-a2b4")
        # a3 cache target must not leak into the a2-only cache list.
        assert json.loads(outputs["csrc_cache_target_ids"]) == ["a2-arm64-ubuntu"]

    def test_no_a2_groups_reports_false(self, dual, tmp_path, monkeypatch):
        github_output = tmp_path / "github_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
        groups = [_make_group("a3", "linux-aarch64-a3-2")]
        assert (
            dual.main(
                [
                    "--test-groups-json",
                    json.dumps(groups),
                    "--a2b1-runner-label",
                    A2B1_LABEL,
                    "--a2b4-runner-label",
                    A2B4_LABEL,
                ]
            )
            == 0
        )
        outputs = dict(line.partition("=")[::2] for line in github_output.read_text().splitlines())
        assert outputs["has_tests_a2"] == "false"
        assert json.loads(outputs["test_groups_a2b1"]) == []
        assert json.loads(outputs["test_groups_a2b4"]) == []


class TestFullSuiteIntegration:
    """End-to-end: real select_tests.py full-suite output -> dual A2 pools."""

    @pytest.fixture(scope="class")
    def full_suite_groups(self):
        pytest.importorskip("regex")
        pytest.importorskip("yaml")
        result = subprocess.run(
            [sys.executable, str(_SELECT_TESTS_PATH), "--all-tests", "--skip-cpu-ut"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("test_groups="):
                return json.loads(line.partition("=")[2])
        pytest.fail("select_tests.py did not emit a test_groups output")

    def test_full_suite_contains_all_chip_partitions(self, full_suite_groups):
        npu_types = {group["npu_type"] for group in full_suite_groups}
        # Sanity: the full suite really routes to a2/a3/310p partitions before
        # the A2 filter narrows it down.
        assert "a2" in npu_types
        assert "a3" in npu_types

    def test_a2_groups_are_one_card_only(self, dual, full_suite_groups):
        a2_groups = dual.filter_a2_groups(full_suite_groups)
        assert a2_groups, "expected one_card A2 groups in the full-suite selection"
        assert all(group["npu_type"] == "a2" for group in a2_groups)
        assert all(group["num_npus"] == 1 for group in a2_groups)
        # 7 load-balanced buckets, exactly like the regular a2 routing.
        assert len(a2_groups) == BUCKET_COUNT
        assert {group["partition"] for group in a2_groups} == {
            f"{i}-{BUCKET_COUNT}" for i in range(1, BUCKET_COUNT + 1)
        }
        tests = [test for group in a2_groups for test in group["tests"].split()]
        assert len(tests) > 50
        # Only one_card tests, and no 310p-specific files (they route to the
        # 310p partition and are filtered out with the rest).
        assert all(test.startswith("tests/e2e/pull_request/one_card") for test in tests)
        assert not any("_310p" in test for test in tests)
        # No duplicates across buckets.
        assert len(tests) == len(set(tests))

    def test_dual_pools_run_identical_suite(self, dual, full_suite_groups):
        a2_groups = dual.filter_a2_groups(full_suite_groups)
        b1_groups = dual.rewrite_runner(a2_groups, A2B1_LABEL, "a2b1")
        b4_groups = dual.rewrite_runner(a2_groups, A2B4_LABEL, "a2b4")

        assert len(b1_groups) == len(b4_groups) == BUCKET_COUNT
        assert {group["runner"] for group in b1_groups} == {A2B1_LABEL}
        assert {group["runner"] for group in b4_groups} == {A2B4_LABEL}
        # Both pools execute the exact same buckets/tests.
        assert [group["tests"] for group in b1_groups] == [group["tests"] for group in b4_groups]
        assert [group["partition"] for group in b1_groups] == [group["partition"] for group in b4_groups]
        # Cache target stays the a2 one so the a2 csrc cache is reused.
        assert dual.collect_csrc_cache_targets(a2_groups) == ["a2-arm64-ubuntu"]
