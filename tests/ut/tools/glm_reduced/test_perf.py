# SPDX-License-Identifier: Apache-2.0
"""Tests for performance aggregation/comparison logic (pure CPU, fail-closed)."""

import json

import pytest

from tools.glm_reduced.perf import PERF_FORMAT, aggregate, compare_perf, load_perf

META_BASE = {
    "model": "/models/glm-reduced",
    "checkpoint_id": "sha256:abc",
    "run_id": "run-A",
    "hardware": "atlas800-a3",
    "workload_id": "abc123",
    "warmup_iterations": 2,
    "measured_iterations": 5,
    "expected_iteration_tokens": 1280,
    "runtime": {"vllm": "0.13.0"},
    "engine": {"tensor_parallel_size": 8, "enable_prefix_caching": False},
}

CANDIDATE_META = {**META_BASE, "run_id": "run-B", "mode": "rev-B", "runtime": {"vllm": "0.14.0"}}
META_BASE["mode"] = "rev-A"


def _write(path, meta, latencies=(1.0, 1.1, 0.9, 1.05, 0.95), tokens=None):
    tokens = tokens or [meta["expected_iteration_tokens"]] * len(latencies)
    payload = {"format": PERF_FORMAT, "meta": meta, "latencies_s": list(latencies), "total_tokens": list(tokens)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_aggregate_values(tmp_path):
    run = load_perf(_write(tmp_path / "b.json", META_BASE))
    stats = aggregate(run)
    assert stats.iterations == 5
    assert stats.min_latency_s == 0.9
    assert stats.max_latency_s == 1.1
    assert stats.median_latency_s == 1.0
    assert stats.mean_throughput_tok_s == pytest.approx(1280 / 1.0, rel=0.05)


def test_compare_pass_and_thresholds(tmp_path):
    baseline = load_perf(_write(tmp_path / "b.json", META_BASE))
    candidate = load_perf(_write(tmp_path / "c.json", CANDIDATE_META, (1.02, 1.0, 0.98, 1.0, 1.0)))
    report = compare_perf(baseline, candidate, max_latency_regression_pct=5.0)
    assert report.ok, report.problems
    assert abs(report.latency_change_pct) < 5.0
    assert not compare_perf(baseline, candidate, max_latency_regression_pct=-1.0).ok


def test_missing_threshold_fails_closed(tmp_path):
    # Independent-review reproduction: a 900% regression must not pass when
    # the caller forgot to supply thresholds.
    baseline = load_perf(_write(tmp_path / "b.json", META_BASE))
    candidate = load_perf(_write(tmp_path / "c.json", CANDIDATE_META, (10.0, 10.0, 10.0, 10.0, 10.0)))
    report = compare_perf(baseline, candidate)
    assert not report.ok
    assert any("threshold" in problem for problem in report.problems)
    # With a threshold, the regression is reported honestly.
    report = compare_perf(baseline, candidate, max_latency_regression_pct=5.0)
    assert not report.ok
    assert report.latency_change_pct > 500


def test_missing_or_invalid_baseline_fails(tmp_path):
    candidate = load_perf(_write(tmp_path / "c.json", CANDIDATE_META))
    with pytest.raises(FileNotFoundError):
        load_perf(tmp_path / "missing.json")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps(
            {
                "format": PERF_FORMAT,
                "meta": META_BASE,
                "latencies_s": [float("nan")] * 5,
                "total_tokens": [1280] * 5,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive finite"):
        load_perf(bad_path)
    # Token-count violation (early EOS / cache-shortened work) fails validation.
    short_path = tmp_path / "short.json"
    _write(short_path, CANDIDATE_META, tokens=[1280] * 4 + [640])
    with pytest.raises(ValueError, match="fixed-workload violation"):
        load_perf(short_path)
    # Workload mismatch with a valid baseline never passes silently.
    baseline = load_perf(_write(tmp_path / "b.json", {**META_BASE, "workload_id": "different"}))
    report = compare_perf(baseline, candidate, max_latency_regression_pct=100.0)
    assert not report.ok
    assert any("workload_id" in problem for problem in report.problems)


def test_checkpoint_identity_rules(tmp_path):
    baseline = load_perf(_write(tmp_path / "b.json", META_BASE))
    # Same checkpoint across runtime revisions is the intended comparison.
    candidate = load_perf(_write(tmp_path / "c.json", CANDIDATE_META))
    assert compare_perf(baseline, candidate, max_latency_regression_pct=100.0).ok
    # Independent A/A runs (identical settings, distinct run_id) are valid.
    aa = load_perf(_write(tmp_path / "aa.json", {**META_BASE, "run_id": "run-A2"}))
    assert compare_perf(baseline, aa, max_latency_regression_pct=100.0).ok
    # The same run (same run_id) is not a comparison.
    same = load_perf(_write(tmp_path / "e.json", dict(META_BASE)))
    report = compare_perf(baseline, same, max_latency_regression_pct=100.0)
    assert not report.ok
    assert any("same run_id" in problem for problem in report.problems)
    # A different checkpoint (e.g. full vs reduced) is not a regression gate.
    other = load_perf(_write(tmp_path / "d.json", {**CANDIDATE_META, "checkpoint_id": "sha256:full"}))
    report = compare_perf(baseline, other, max_latency_regression_pct=100.0)
    assert not report.ok
    assert any("checkpoint_id mismatch" in problem for problem in report.problems)


def test_tp_mismatch_rejected_for_perf(tmp_path):
    # Independent-review reproduction: tp=1 vs tp=8 must not compare as a
    # valid regression; engine settings must be identical for perf.
    baseline = load_perf(_write(tmp_path / "b.json", META_BASE))
    tp1 = load_perf(
        _write(
            tmp_path / "tp1.json",
            {**CANDIDATE_META, "engine": {"tensor_parallel_size": 1, "enable_prefix_caching": False}},
        )
    )
    report = compare_perf(baseline, tp1, max_latency_regression_pct=100.0)
    assert not report.ok
    assert any("engine settings mismatch" in problem for problem in report.problems)
    dtype = load_perf(_write(tmp_path / "dt.json", {**CANDIDATE_META, "dtype": "fp8"}))
    report = compare_perf(baseline, dtype, max_latency_regression_pct=100.0)
    assert not report.ok
    assert any("dtype mismatch" in problem for problem in report.problems)


def test_invalid_sample_counts_rejected(tmp_path):
    meta = {**CANDIDATE_META, "measured_iterations": 0}
    path = tmp_path / "zero.json"
    path.write_text(
        json.dumps({"format": PERF_FORMAT, "meta": meta, "latencies_s": [], "total_tokens": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive integer"):
        load_perf(path)
    meta = {**CANDIDATE_META, "runtime": {}}
    path = tmp_path / "nort.json"
    _write(path, meta)
    with pytest.raises(ValueError, match="runtime identity"):
        load_perf(path)


def test_sample_count_must_match_declared_iterations(tmp_path):
    path = tmp_path / "short.json"
    path.write_text(
        json.dumps({"format": PERF_FORMAT, "meta": META_BASE, "latencies_s": [1.0], "total_tokens": [1280]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="measured_iterations"):
        load_perf(path)
