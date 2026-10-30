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

"""Tests for single-card operator E2E runner assignment."""

from __future__ import annotations

import json
import sys

import pytest
import select_tests

_OPS_TEST = "tests/e2e/nightly/single_node/ops/singlecard_ops/test_grouped_matmul_situ_quant_weight_nz.py"
_A3_E2E = "tests/e2e/pull_request/four_card/test_data_parallel_tp2.py"
_A5_E2E = "tests/e2e/pull_request/one_card/test_flash_attn_precision.py"


def _groups_from_stdout(capsys: pytest.CaptureFixture[str]) -> list[dict]:
    output = capsys.readouterr().out
    for line in output.splitlines():
        if line.startswith("test_groups="):
            return json.loads(line.split("=", 1)[1])
    raise AssertionError(f"test_groups missing from output:\n{output}")


def _runners_for(groups: list[dict], test_path: str) -> set[str]:
    return {group["runner"] for group in groups if test_path in group["tests"].split()}


def test_recommended_operator_test_joins_single_card_runners(monkeypatch, capsys, tmp_path):
    list_file = tmp_path / "recommended.txt"
    list_file.write_text(f"{_OPS_TEST}\n{_A3_E2E}\n{_A5_E2E}\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_tests.py", "--test-list-file", str(list_file), "--skip-cpu-ut"],
    )

    select_tests.main()

    groups = _groups_from_stdout(capsys)
    assert _runners_for(groups, _OPS_TEST) == {
        "linux-aarch64-a3-1",
        "linux-aarch64-a5-1",
    }
    assert "linux-aarch64-a3-4" in _runners_for(groups, _A3_E2E)
    assert _runners_for(groups, _A5_E2E) == {"linux-aarch64-a5-4"}


def test_all_tests_skips_operator_directory(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(sys, "argv", ["select_tests.py", "--all-tests", "--skip-cpu-ut"])

    select_tests.main()

    groups = _groups_from_stdout(capsys)
    assert _runners_for(groups, _OPS_TEST) == set()


def test_explicit_e2e_does_not_schedule_operator_directory(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["select_tests.py", "--explicit-e2e-tests", _OPS_TEST],
    )

    select_tests.main()

    groups = _groups_from_stdout(capsys)
    assert groups == []


def test_operator_path_is_kept_on_every_single_card_partition():
    target = "tests/e2e/nightly/single_node/ops/singlecard_ops/test_op.py"
    groups = {
        "a3-1": [target, target, "tests/e2e/pull_request/four_card/test_a.py"],
        "a5-1": [f"{target}::test_case", target],
        "a2-1": ["tests/e2e/pull_request/one_card/test_b.py", "tests/e2e/pull_request/one_card/test_b.py"],
    }

    select_tests._dedup_groups(groups, "tests/e2e/nightly/single_node/ops/singlecard_ops")

    assert groups["a3-1"] == [target, "tests/e2e/pull_request/four_card/test_a.py"]
    assert groups["a5-1"] == [target]
    assert groups["a2-1"] == ["tests/e2e/pull_request/one_card/test_b.py"]
