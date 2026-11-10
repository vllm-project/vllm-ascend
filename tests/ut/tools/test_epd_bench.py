#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/epd_bench.py (pure functions, no NPU/vllm)."""

from __future__ import annotations

import json

import pytest

from tools.epd_bench import (
    aggregate_stage_metrics,
    compare_with_tolerance,
    main,
    parse_bench_output,
    render_json,
    render_markdown,
)

SERVING_OUTPUT = """============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Benchmark duration (s):                  19.92
Total input tokens:                      1374
Total generated tokens:                  2663
Request throughput (req/s):              0.50
Output token throughput (tok/s):         133.67
Total Token throughput (tok/s):          202.64
---------------Time to First Token----------------
Mean TTFT (ms):                          127.10
Median TTFT (ms):                        136.29
P99 TTFT (ms):                           137.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.85
Median TPOT (ms):                        25.78
P99 TPOT (ms):                           26.64
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.78
Median ITL (ms):                         25.74
P99 ITL (ms):                            28.85
==================================================
"""


def test_parse_bench_output() -> None:
    parsed = parse_bench_output(SERVING_OUTPUT)

    assert parsed["successful_requests"] == 10
    assert parsed["failed_requests"] == 0
    assert parsed["benchmark_duration_s"] == pytest.approx(19.92)
    assert parsed["total_input_tokens"] == 1374
    assert parsed["total_generated_tokens"] == 2663
    assert parsed["request_throughput_rps"] == pytest.approx(0.50)
    assert parsed["output_token_throughput"] == pytest.approx(133.67)
    assert parsed["total_token_throughput"] == pytest.approx(202.64)
    assert parsed["mean_ttft_ms"] == pytest.approx(127.10)
    assert parsed["median_ttft_ms"] == pytest.approx(136.29)
    assert parsed["p99_ttft_ms"] == pytest.approx(137.83)
    assert parsed["mean_tpot_ms"] == pytest.approx(25.85)
    assert parsed["p99_tpot_ms"] == pytest.approx(26.64)
    assert parsed["mean_itl_ms"] == pytest.approx(25.78)

    offline = parse_bench_output("Throughput: 2.73 requests/s, 699.93 total tokens/s, 349.97 output tokens/s")
    assert offline["request_throughput_rps"] == pytest.approx(2.73)
    assert offline["total_token_throughput"] == pytest.approx(699.93)
    assert offline["output_token_throughput"] == pytest.approx(349.97)

    missing = parse_bench_output("")
    assert all(value is None for value in missing.values()), "empty output must not invent values"


def test_aggregate_stage_metrics() -> None:
    rows = [
        {"stage": "encode", "seconds": 0.2},
        {"stage": "encode", "seconds": 0.4},
        {"stage": "encode", "seconds": 0.6},
        {"stage": "decode", "seconds": 0.01},
        {"stage": "unknown", "seconds": None},
        "not-a-row",
    ]
    aggregated = aggregate_stage_metrics(rows)

    encode = aggregated["encode"]
    assert encode["count"] == 3
    assert encode["mean_ms"] == pytest.approx(400.0)
    assert encode["p50_ms"] == pytest.approx(400.0)
    assert encode["p99_ms"] == pytest.approx(600.0)
    assert encode["min_ms"] == pytest.approx(200.0)
    assert encode["max_ms"] == pytest.approx(600.0)

    decode = aggregated["decode"]
    assert decode["count"] == 1
    assert decode["mean_ms"] == pytest.approx(10.0)
    assert decode["p50_ms"] == pytest.approx(10.0)
    assert decode["p99_ms"] == pytest.approx(10.0)

    assert aggregate_stage_metrics([]) == {}


def _baseline() -> dict:
    return {
        "benchmark": {
            "request_throughput_rps": 100.0,
            "output_token_throughput": 1000.0,
            "mean_ttft_ms": 100.0,
            "mean_tpot_ms": 50.0,
        }
    }


def _current(**overrides: float) -> dict:
    metrics = dict(_baseline()["benchmark"])
    metrics.update(overrides)
    return {"benchmark": metrics}


def test_compare_with_tolerance() -> None:
    within = compare_with_tolerance(
        _current(request_throughput_rps=97.0, mean_ttft_ms=97.0, mean_tpot_ms=51.5),
        _baseline(),
        tol=0.05,
    )
    statuses = {entry["metric"]: entry["status"] for entry in within["metrics"]}
    assert statuses["request_throughput_rps"] == "within_tolerance"  # -3% vs 5% tolerance
    assert statuses["mean_ttft_ms"] == "within_tolerance"  # lower latency is better, -3% is below 5%
    assert statuses["mean_tpot_ms"] == "within_tolerance"  # +3% vs 5%
    assert within["has_regression"] is False

    regressed = compare_with_tolerance(
        _current(request_throughput_rps=94.0, mean_tpot_ms=53.0),
        _baseline(),
        tol=0.05,
    )
    assert set(regressed["regressions"]) == {"request_throughput_rps", "mean_tpot_ms"}
    assert regressed["has_regression"] is True

    improved = compare_with_tolerance(
        _current(request_throughput_rps=110.0, mean_ttft_ms=90.0),
        _baseline(),
        tol=0.05,
    )
    assert set(improved["improvements"]) == {"request_throughput_rps", "mean_ttft_ms"}
    assert improved["has_regression"] is False

    boundary = compare_with_tolerance(
        _current(request_throughput_rps=95.0, mean_ttft_ms=105.0),
        _baseline(),
        tol=0.05,
    )
    boundary_statuses = {entry["metric"]: entry["status"] for entry in boundary["metrics"]}
    assert boundary_statuses["request_throughput_rps"] == "within_tolerance"  # exactly -5%
    assert boundary_statuses["mean_ttft_ms"] == "within_tolerance"  # exactly +5%
    assert boundary["has_regression"] is False

    missing = compare_with_tolerance({"benchmark": {"request_throughput_rps": 100.0}}, _baseline(), tol=0.05)
    missing_statuses = {entry["metric"]: entry["status"] for entry in missing["metrics"]}
    assert missing_statuses["mean_ttft_ms"] == "missing"
    assert missing["has_regression"] is False


def test_report_markdown_contains_required_fields() -> None:
    benchmark = parse_bench_output(SERVING_OUTPUT)
    report = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "command": ["vllm", "bench", "serve"],
        "benchmark": benchmark,
        "stages": aggregate_stage_metrics([{"stage": "encode", "seconds": 0.01}]),
        "ec_transfer_bytes": 1024,
        "hbm_peak_gb": None,
        "comparison": compare_with_tolerance(_current(request_throughput_rps=90.0), _baseline()),
        "notes": ["synthetic test data"],
    }
    markdown = render_markdown(report)

    required = [
        "Request throughput (req/s)",
        "Output token throughput (tok/s)",
        "Total token throughput (tok/s)",
        "Mean TTFT (ms)",
        "Mean TPOT (ms)",
        "encode",
        "EC transfer bytes",
        "HBM peak (GB)",
        "regressed",
    ]
    for field in required:
        assert field in markdown, f"report markdown is missing {field!r}"
    assert "未采集" in markdown, "missing HBM peak must be rendered as 未采集"
    assert "None" not in markdown, "the renderer must not print Python None"

    empty = render_markdown({"model": "m", "benchmark": parse_bench_output(""), "notes": []})
    assert "未采集" in empty
    assert "None" not in empty

    parsed = json.loads(render_json(report))
    assert parsed["benchmark"]["mean_ttft_ms"] == pytest.approx(127.10)
    assert parsed["stages"]["encode"]["count"] == 1


def test_help_works_without_vllm() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
