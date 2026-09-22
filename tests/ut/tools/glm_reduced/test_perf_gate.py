# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the GLM5.x 11-layer performance gate.

Synthetic numbers here are gate-contract evidence, never performance evidence.
"""

import copy

import pytest

from tools.glm_reduced.perf_gate import compare, load_baseline, validate, workload_id


def fixture():
    return copy.deepcopy(load_baseline())


def candidate(baseline, throughput=1.0, latency=1.0):
    record = copy.deepcopy(baseline)
    for run in record["runs"]:
        aggregate = run["aggregate"]
        aggregate["output_tokens_per_second"]["median"] *= throughput
        for metric in ("ttft_seconds", "tpot_seconds", "seconds"):
            aggregate[metric]["median"] *= latency
    return record


def test_pinned_baseline_is_valid():
    baseline = fixture()
    assert validate(baseline) == []
    assert [workload_id(run) for run in baseline["runs"]] == ["128x128x1", "4096x128x8", "128x1024x8"]


def test_thresholds_are_required():
    report = compare(fixture(), fixture())
    assert report["ok"] is False
    assert any("threshold" in problem for problem in report["problems"])


def test_pass_within_tolerance():
    baseline = fixture()
    report = compare(
        baseline,
        candidate(baseline, throughput=0.96, latency=1.08),
        min_throughput_fraction=0.95,
        max_latency_fraction=1.10,
    )
    assert report["ok"] is True, report["problems"]
    assert report["workloads"]["128x128x1"]["throughput_fraction"] == pytest.approx(0.96)


def test_throughput_regression_fails():
    baseline = fixture()
    report = compare(
        baseline,
        candidate(baseline, throughput=0.94, latency=1.0),
        min_throughput_fraction=0.95,
        max_latency_fraction=1.10,
    )
    assert report["ok"] is False
    assert any("throughput" in problem for problem in report["problems"])


def test_latency_regression_fails():
    baseline = fixture()
    report = compare(
        baseline,
        candidate(baseline, throughput=1.0, latency=1.15),
        min_throughput_fraction=0.95,
        max_latency_fraction=1.10,
    )
    assert report["ok"] is False
    assert any("ttft_seconds" in problem or "tpot_seconds" in problem for problem in report["problems"])


@pytest.mark.parametrize("mutation", ["hardware", "engine", "workloads", "warmup"])
def test_identity_mismatch_fails_closed(mutation):
    baseline = fixture()
    probe = fixture()
    if mutation == "hardware":
        probe["identity"]["hardware"] = "linux-aarch64-a3-800i-8"
    elif mutation == "engine":
        probe["identity"]["engine_settings"]["max_num_seqs"] = 8
    elif mutation == "workloads":
        probe["identity"]["workloads"] = [[128, 128, 1]]
    else:
        probe["identity"]["warmup_iterations"] = 2
    report = compare(baseline, probe, min_throughput_fraction=0.95, max_latency_fraction=1.10)
    assert report["ok"] is False


@pytest.mark.parametrize("mutation", ["missing_metric", "zero", "nan", "duplicate", "no_runs"])
def test_malformed_records_fail_closed(mutation):
    baseline = fixture()
    probe = fixture()
    if mutation == "missing_metric":
        probe["runs"][0]["aggregate"].pop("tpot_seconds")
    elif mutation == "zero":
        probe["runs"][0]["aggregate"]["seconds"]["median"] = 0
    elif mutation == "nan":
        probe["runs"][0]["aggregate"]["ttft_seconds"]["median"] = float("nan")
    elif mutation == "duplicate":
        probe["runs"].append(copy.deepcopy(probe["runs"][0]))
    else:
        probe["runs"] = []
    report = compare(baseline, probe, min_throughput_fraction=0.95, max_latency_fraction=1.10)
    assert report["ok"] is False


def test_candidate_workload_absent_from_baseline_fails():
    baseline = fixture()
    probe = fixture()
    extra = copy.deepcopy(probe["runs"][0])
    extra["input_tokens"] = 2048
    probe["runs"].append(extra)
    report = compare(baseline, probe, min_throughput_fraction=0.95, max_latency_fraction=1.10)
    assert report["ok"] is False
    assert any("absent from the baseline" in problem for problem in report["problems"])
