# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
import pytest

from tests.e2e.nightly.multi_node.scripts.benchmark_results import (
    compare_version_results,
    get_output_throughput,
)


def _perf_result(throughput: float) -> list:
    return [None, {"Output Token Throughput": {"total": f"{throughput} token/s"}}]


def test_get_output_throughput() -> None:
    assert get_output_throughput(_perf_result(123.4)) == pytest.approx(123.4)
    assert get_output_throughput([None, {}]) is None
    assert get_output_throughput(None) is None
    assert get_output_throughput("unexpected") is None


def test_compare_version_results_passes() -> None:
    cases = [
        {"case_name": "perf_128k", "case_type": "performance", "dataset_path": "ds1", "version_threshold": 0.97},
        {"case_name": "perf_16k", "case_type": "performance", "dataset_path": "ds2", "version_threshold": 0.97},
    ]
    results = {
        "v2": {"perf_128k": _perf_result(100.0), "perf_16k": _perf_result(200.0)},
        "v1": {"perf_128k": _perf_result(99.0), "perf_16k": _perf_result(205.0)},
    }
    report, passed = compare_version_results(cases, results, "v1", default_threshold=0.97)
    assert passed
    assert len(report) == 2
    by_case = {entry["case_name"]: entry for entry in report}
    assert by_case["perf_128k"]["ratio"] == pytest.approx(round(100.0 / 99.0, 4), abs=1e-9)
    assert by_case["perf_16k"]["ratio"] == pytest.approx(round(200.0 / 205.0, 4), abs=1e-9)
    assert all(entry["passed"] for entry in report)


def test_compare_version_results_fails_below_threshold() -> None:
    cases = [{"case_name": "perf_16k", "case_type": "performance", "version_threshold": 0.97}]
    results = {
        "v2": {"perf_16k": _perf_result(90.0)},
        "v1": {"perf_16k": _perf_result(100.0)},
    }
    report, passed = compare_version_results(cases, results, "v1")
    assert not passed
    assert report[0]["ratio"] == pytest.approx(0.9)
    assert report[0]["passed"] is False


def test_compare_version_results_skips_non_perf_and_uses_default_threshold() -> None:
    cases = [
        {"case_name": "acc", "case_type": "accuracy"},
        {"case_name": "perf", "case_type": "performance"},
    ]
    results = {
        "v2": {"perf": _perf_result(95.0)},
        "v1": {"perf": _perf_result(100.0)},
    }
    report, passed = compare_version_results(cases, results, "v1", default_threshold=0.9)
    assert passed
    assert len(report) == 1
    assert report[0]["case_name"] == "perf"
    assert report[0]["threshold"] == 0.9


def test_compare_version_results_missing_baseline_fails() -> None:
    cases = [{"case_name": "perf", "case_type": "performance"}]
    report, passed = compare_version_results(cases, {"v2": {}}, "v1")
    assert not passed
    assert "error" in report[0]


def test_compare_version_results_missing_candidate_fails() -> None:
    cases = [{"case_name": "perf", "case_type": "performance", "version_threshold": 0.97}]
    results = {
        "v2": {},
        "v1": {"perf": _perf_result(100.0)},
    }
    report, passed = compare_version_results(cases, results, "v1")
    assert not passed
    assert report[0]["candidate_output_throughput"] is None
