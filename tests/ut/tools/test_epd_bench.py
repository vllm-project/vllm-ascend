#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/epd_bench.py (pure functions, no NPU/vllm)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.epd_bench import (
    _fetch_text,
    _load_ec_bytes,
    _write_reports,
    aggregate_stage_metrics,
    compare_with_tolerance,
    main,
    parse_bench_output,
    parse_proxy_metrics,
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


class _UnreadableCacheEntry:
    """Stand-in for an encoder-cache entry whose stat() fails mid-scan."""

    def is_file(self) -> bool:
        return True

    def stat(self) -> object:
        raise OSError("simulated stat failure")


def test_fetch_text_rejects_url_without_host() -> None:
    with pytest.raises(ValueError) as excinfo:
        _fetch_text("not-a-url")
    assert "has no host" in str(excinfo.value)
    assert "not-a-url" in str(excinfo.value)


def test_fetch_text_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="only http and https are supported"):
        _fetch_text("ftp://127.0.0.1:9100/metrics")


def test_load_ec_bytes_returns_none_when_stat_fails(monkeypatch, tmp_path) -> None:
    (tmp_path / "entry.safetensors").write_bytes(b"x")
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter([_UnreadableCacheEntry()]))
    args = SimpleNamespace(ec_transfer_bytes=None, ec_shared_storage_path=str(tmp_path))
    assert _load_ec_bytes(args) is None


def test_main_reports_missing_vllm_as_user_error(monkeypatch, tmp_path) -> None:
    report_md = tmp_path / "epd_report.md"
    report_json = tmp_path / "epd_report.json"

    def _missing_vllm(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("vllm")

    monkeypatch.setattr(subprocess, "run", _missing_vllm)
    exit_code = main(["--model", "m", "--report-md", str(report_md), "--report-json", str(report_json)])

    assert exit_code == 2
    assert report_md.is_file()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert any("failed to execute benchmark command" in note for note in report["notes"])
    assert report["benchmark"] == parse_bench_output("")


SUMMARY_ONLY_METRICS = """
epd_stage_seconds_count{stage="encode"} 4
epd_stage_seconds_sum{stage="encode"} 0.096
"""

BUCKET_METRICS = """
epd_stage_seconds_bucket{stage="encode",le="0.05"} 2
epd_stage_seconds_bucket{stage="encode",le="0.1"} 4
epd_stage_seconds_bucket{stage="encode",le="+Inf"} 4
epd_stage_seconds_count{stage="encode"} 4
epd_stage_seconds_sum{stage="encode"} 0.096
"""


def test_parse_proxy_metrics_reports_bucket_upper_bounds() -> None:
    stages = parse_proxy_metrics(BUCKET_METRICS)

    encode = stages["encode"]
    assert encode["count"] == 4
    assert encode["mean_ms"] == pytest.approx(24.0)
    # The percentiles are histogram bucket upper bounds, not exact quantiles.
    assert encode["p50_ms"] == pytest.approx(50.0)
    assert encode["p99_ms"] == pytest.approx(100.0)


def test_parse_proxy_metrics_never_falls_back_to_the_mean() -> None:
    stages = parse_proxy_metrics(SUMMARY_ONLY_METRICS)

    encode = stages["encode"]
    assert encode["mean_ms"] == pytest.approx(24.0)
    assert encode["p50_ms"] is None, "a summary-only exporter must not turn the mean into p50"
    assert encode["p99_ms"] is None, "a summary-only exporter must not turn the mean into p99"


def test_parse_proxy_metrics_requires_the_inf_bucket_for_percentiles() -> None:
    without_inf = "\n".join(line for line in BUCKET_METRICS.splitlines() if "+Inf" not in line)
    encode = parse_proxy_metrics(without_inf)["encode"]
    assert encode["count"] == 4
    assert encode["p50_ms"] is None
    assert encode["p99_ms"] is None

    # Exporters may write integral counts as floats; the parser still reads them.
    float_counts = BUCKET_METRICS.replace("} 4", "} 4.0").replace("} 2", "} 2.0")
    encode = parse_proxy_metrics(float_counts)["encode"]
    assert encode["p50_ms"] == pytest.approx(50.0)
    assert encode["p99_ms"] == pytest.approx(100.0)

    empty = parse_proxy_metrics("")
    assert empty == {}


def test_parse_proxy_metrics_handles_a_partial_family() -> None:
    buckets_only = "\n".join(
        line for line in BUCKET_METRICS.splitlines() if "count{" not in line and "sum{" not in line
    )
    encode = parse_proxy_metrics(buckets_only)["encode"]
    assert encode["count"] == 4, "the +Inf bucket is the observation count"
    assert encode["mean_ms"] is None, "a missing _sum must stay unmeasured"
    assert encode["p99_ms"] == pytest.approx(100.0)


def test_parse_bench_output_keeps_a_measured_zero() -> None:
    text = (
        "Request throughput (req/s): 0.00\nThroughput: 2.73 requests/s, 699.93 total tokens/s, 349.97 output tokens/s"
    )
    parsed = parse_bench_output(text)
    assert parsed["request_throughput_rps"] == pytest.approx(0.0)


def test_dry_run_records_the_command_without_running_it(tmp_path) -> None:
    report_md = tmp_path / "epd_report.md"
    report_json = tmp_path / "epd_report.json"

    exit_code = main(["--model", "m", "--dry-run", "--report-md", str(report_md), "--report-json", str(report_json)])

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["command"][:3] == ["vllm", "bench", "serve"]
    assert report["benchmark_executed"] is False
    assert any("not executed" in note for note in report["notes"])
    assert all(value is None for value in report["benchmark"].values())
    assert not list(tmp_path.glob("*.tmp")), "no temporary report file may be left behind"
    markdown = report_md.read_text(encoding="utf-8")
    assert "- Benchmark command (not executed): vllm bench serve" in markdown


def test_bench_output_does_not_record_a_command(tmp_path) -> None:
    bench_output = tmp_path / "bench.txt"
    bench_output.write_text(SERVING_OUTPUT, encoding="utf-8")
    report_json = tmp_path / "epd_report.json"

    exit_code = main(
        [
            "--model",
            "m",
            "--bench-output",
            str(bench_output),
            "--report-md",
            str(tmp_path / "r.md"),
            "--report-json",
            str(report_json),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["command"] is None, "a parsed previous run must not claim a command was executed"
    assert report["benchmark_executed"] is False
    assert any("no benchmark command was executed" in note for note in report["notes"])
    assert report["benchmark"]["successful_requests"] == 10


def _run_cli(tmp_path, *extra: str) -> tuple[int, Path, Path]:
    report_md = tmp_path / "epd_report.md"
    report_json = tmp_path / "epd_report.json"
    args = ["--model", "m", "--report-md", str(report_md), "--report-json", str(report_json), *extra]
    return main(args), report_md, report_json


def test_invalid_inputs_are_user_errors(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")

    cases = [
        (["--bench-output", str(tmp_path / "missing.txt")], "does not exist"),
        (["--dry-run", "--baseline-json", str(tmp_path / "missing.json")], "does not exist"),
        (["--dry-run", "--fail-on-regression"], "needs --baseline-json"),
        (["--dry-run", "--tolerance", "0"], "tolerance"),
        (["--dry-run", "--tolerance", "1.5"], "tolerance"),
        (["--dry-run", "--bench-output", str(baseline)], "cannot be combined"),
        (["--dry-run", "--report-md", str(tmp_path / "missing" / "r.md")], "parent directory"),
    ]
    for extra, expected in cases:
        exit_code, report_md, report_json = _run_cli(tmp_path, *extra)
        assert exit_code == 2, f"{extra} should be rejected as a user error"
        assert expected in capsys.readouterr().err
        assert not report_md.exists() and not report_json.exists(), f"{extra} must not write a report"


def test_broken_baseline_still_writes_the_report(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not-json", encoding="utf-8")

    exit_code, report_md, report_json = _run_cli(tmp_path, "--dry-run", "--baseline-json", str(baseline))

    assert exit_code == 0
    assert report_md.is_file() and report_json.is_file()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["comparison"] is None
    assert any("comparison skipped" in note for note in report["notes"])


def test_benchmark_timeout_is_a_user_error(monkeypatch, tmp_path) -> None:
    def _timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="vllm", timeout=1, output="partial stdout\n")

    monkeypatch.setattr(subprocess, "run", _timeout)

    exit_code, report_md, report_json = _run_cli(tmp_path, "--bench-timeout-seconds", "1")

    assert exit_code == 2
    assert report_md.is_file()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert any("timed out" in note for note in report["notes"])
    assert any("partial stdout" in note for note in report["notes"])


def test_reports_are_written_as_a_pair(monkeypatch, tmp_path) -> None:
    reports = SimpleNamespace(report_md=str(tmp_path / "r.md"), report_json=str(tmp_path / "r.json"))
    original_write_text = Path.write_text
    calls = {"count": 0}

    def _flaky_write_text(self, data, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated disk failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _flaky_write_text)
    with pytest.raises(OSError, match="simulated disk failure"):
        _write_reports(reports, {"model": "m", "benchmark": parse_bench_output("")})

    assert not (tmp_path / "r.md").exists(), "the markdown half must not be committed alone"
    assert not (tmp_path / "r.json").exists()
    assert not list(tmp_path.glob("*.tmp")), "temporary files must be cleaned up"


def test_executed_benchmark_is_marked_as_executed(monkeypatch, tmp_path) -> None:
    completed = SimpleNamespace(returncode=0, stdout=SERVING_OUTPUT, stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    exit_code, report_md, report_json = _run_cli(tmp_path)

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["benchmark_executed"] is True
    markdown = report_md.read_text(encoding="utf-8")
    assert "- Benchmark command: vllm bench serve" in markdown
    assert "(not executed)" not in markdown


def test_proxy_percentiles_are_annotated_as_bucket_bounds(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.txt"
    metrics_path.write_text(BUCKET_METRICS, encoding="utf-8")
    report = {
        "model": "m",
        "stages": parse_proxy_metrics(BUCKET_METRICS),
        "stage_percentile_source": "prometheus_bucket_upper_bound",
        "notes": [],
    }
    markdown = render_markdown(report)
    assert "histogram bucket upper bounds, not exact quantiles" in markdown


def test_stage_metrics_json_reports_exact_percentiles(tmp_path) -> None:
    stage_rows = tmp_path / "stages.json"
    stage_rows.write_text(json.dumps([{"stage": "encode", "seconds": 0.01}]), encoding="utf-8")

    exit_code, report_md, report_json = _run_cli(tmp_path, "--dry-run", "--stage-metrics-json", str(stage_rows))

    assert exit_code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["stage_percentile_source"] == "exact"
    markdown = report_md.read_text(encoding="utf-8")
    assert "bucket upper bounds" not in markdown


def test_help_documents_the_exit_codes(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    assert "Exit codes" in capsys.readouterr().out


def test_stage_rows_report_unmeasured_percentiles() -> None:
    report = {
        "model": "m",
        "benchmark": parse_bench_output(""),
        "stages": parse_proxy_metrics(SUMMARY_ONLY_METRICS),
        "notes": [],
    }
    markdown = render_markdown(report)
    row = next(line for line in markdown.splitlines() if line.startswith("| encode |"))

    assert "| 4 | 24.0000 | 未采集 | 未采集 | 未采集 |" in row
