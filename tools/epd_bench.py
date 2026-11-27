#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""EPD benchmark report harness.

The module only imports the standard library at import time, so
```python tools/epd_bench.py --help``` works on a CPU-only machine without
vllm installed. Actual benchmarking is delegated to the repository-standard
```vllm bench serve``` command; this tool parses its text output, optionally
aggregates EPD proxy stage metrics, compares against a baseline report, and
renders a markdown + JSON report.

Reporting rules:

- A value that was not measured renders as "未采集". A missing histogram, a
  missing +Inf bucket or an unparsable field is never replaced by a mean.
- P50/P99 parsed from a Prometheus endpoint are histogram bucket upper bounds,
  not exact quantiles, and the report states that next to the table.
- ec_transfer_bytes collected from --ec-shared-storage-path is the sum of the
  *.safetensors file sizes at report time, not the bytes moved by this run.

Exit codes: 0 report written; 1 regression exceeded --tolerance with
--fail-on-regression; 2 the harness could not produce a trustworthy report
(bad input path, benchmark failure or timeout, unreachable report directory).
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE = 0.05
DEFAULT_BENCH_TIMEOUT_SECONDS = 7200.0
UNMEASURED = "未采集"
MEASURED = "实测"
MILLISECONDS_PER_SECOND = 1000.0
OUTPUT_TAIL_CHARS = 1000
ERROR_EXIT_CODE = 2
REGRESSION_EXIT_CODE = 1
STAGE_PERCENTILE_EXACT = "exact"
STAGE_PERCENTILE_BUCKET = "prometheus_bucket_upper_bound"

BENCHMARK_KEYS = (
    "successful_requests",
    "failed_requests",
    "benchmark_duration_s",
    "total_input_tokens",
    "total_generated_tokens",
    "request_throughput_rps",
    "output_token_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)
INT_KEYS = frozenset(
    {
        "successful_requests",
        "failed_requests",
        "total_input_tokens",
        "total_generated_tokens",
    }
)

COMPARE_METRICS = (
    "request_throughput_rps",
    "output_token_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
)
HIGHER_IS_BETTER = frozenset(
    {
        "request_throughput_rps",
        "output_token_throughput",
        "total_token_throughput",
    }
)
LOWER_IS_BETTER = frozenset({"mean_ttft_ms", "mean_tpot_ms"})

# Aliases used by vllm bench serve --save-result JSON files.
VLLM_RESULT_ALIASES = {
    "request_throughput": "request_throughput_rps",
    "output_throughput": "output_token_throughput",
    "total_token_throughput": "total_token_throughput",
    "mean_ttft_ms": "mean_ttft_ms",
    "mean_tpot_ms": "mean_tpot_ms",
}


def _field_pattern(label: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(label)}:\s*([-+]?[\d,]+(?:\.\d+)?)", re.MULTILINE)


_TEXT_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "successful_requests": _field_pattern("Successful requests"),
    "failed_requests": _field_pattern("Failed requests"),
    "benchmark_duration_s": _field_pattern("Benchmark duration (s)"),
    "total_input_tokens": _field_pattern("Total input tokens"),
    "total_generated_tokens": _field_pattern("Total generated tokens"),
    "request_throughput_rps": _field_pattern("Request throughput (req/s)"),
    "output_token_throughput": _field_pattern("Output token throughput (tok/s)"),
    "total_token_throughput": _field_pattern("Total Token throughput (tok/s)"),
    "mean_ttft_ms": _field_pattern("Mean TTFT (ms)"),
    "median_ttft_ms": _field_pattern("Median TTFT (ms)"),
    "p99_ttft_ms": _field_pattern("P99 TTFT (ms)"),
    "mean_tpot_ms": _field_pattern("Mean TPOT (ms)"),
    "median_tpot_ms": _field_pattern("Median TPOT (ms)"),
    "p99_tpot_ms": _field_pattern("P99 TPOT (ms)"),
    "mean_itl_ms": _field_pattern("Mean ITL (ms)"),
    "median_itl_ms": _field_pattern("Median ITL (ms)"),
    "p99_itl_ms": _field_pattern("P99 ITL (ms)"),
}

_OFFLINE_THROUGHPUT_PATTERN = re.compile(
    r"^Throughput:\s*([\d.,]+) requests/s,\s*([\d.,]+) total tokens/s,\s*([\d.,]+) output tokens/s",
    re.MULTILINE,
)

_STAGE_TOTAL_PATTERN = re.compile(
    r'^epd_stage_seconds_(sum|count)\{stage="([^"]+)"\}\s+([-+]?[\d.eE]+)$',
    re.MULTILINE,
)
_STAGE_BUCKET_PATTERN = re.compile(
    r'^epd_stage_seconds_bucket\{stage="([^"]+)",le="([^"]+)"\}\s+(-?[\d.]+(?:[eE][+-]?\d+)?)$',
    re.MULTILINE,
)


def _to_number(raw: str, integer: bool) -> int | float:
    value = float(raw.replace(",", ""))
    return int(value) if integer else value


def parse_bench_output(text: str) -> dict[str, int | float | None]:
    """Parse the text output of vllm bench serve (or vllm bench throughput)."""
    result: dict[str, int | float | None] = {key: None for key in BENCHMARK_KEYS}
    for key, pattern in _TEXT_FIELD_PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            result[key] = _to_number(match.group(1), key in INT_KEYS)

    offline = _OFFLINE_THROUGHPUT_PATTERN.search(text)
    if offline is not None:
        # Keep a serving number even when it is 0.0; only fill the gaps.
        offline_values = {
            "request_throughput_rps": _to_number(offline.group(1), integer=False),
            "total_token_throughput": _to_number(offline.group(2), integer=False),
            "output_token_throughput": _to_number(offline.group(3), integer=False),
        }
        for key, value in offline_values.items():
            if result[key] is None:
                result[key] = value
    return result


def _percentile_value(ordered_values: list[float], quantile: float) -> float:
    """Nearest-rank percentile; deterministic and easy to hand-check."""
    index = max(0, math.ceil(quantile * len(ordered_values)) - 1)
    return ordered_values[min(index, len(ordered_values) - 1)]


def _stage_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean_ms": sum(ordered) / count * MILLISECONDS_PER_SECOND,
        "p50_ms": _percentile_value(ordered, 0.50) * MILLISECONDS_PER_SECOND,
        "p99_ms": _percentile_value(ordered, 0.99) * MILLISECONDS_PER_SECOND,
        "min_ms": ordered[0] * MILLISECONDS_PER_SECOND,
        "max_ms": ordered[-1] * MILLISECONDS_PER_SECOND,
    }


def aggregate_stage_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate per-request stage durations (rows of {stage, seconds}) in ms."""
    collected: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        stage = row.get("stage")
        seconds = row.get("seconds")
        if not isinstance(stage, str) or not stage or seconds is None:
            continue
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            continue
        collected.setdefault(stage, []).append(value)
    return {stage: _stage_summary(values) for stage, values in sorted(collected.items())}


def _bucket_percentile(
    buckets: list[tuple[float | None, int]],
    quantile: float,
) -> float | None:
    """Nearest-rank percentile from cumulative Prometheus buckets.

    The result is the upper bound of the bucket that contains the quantile,
    not an exact quantile, because /metrics never exposes the individual
    observations. None (rendered as unmeasured) means the value cannot be
    derived: no +Inf bucket to read the observation count from, an empty
    histogram, or a quantile that falls above the last finite bound.
    """
    total = next((cumulative for bound, cumulative in buckets if bound is None), 0)
    if not total:
        return None
    limit = math.ceil(quantile * total)
    for bound, cumulative in buckets:
        if cumulative >= limit:
            return None if bound is None else bound * MILLISECONDS_PER_SECOND
    return None


def parse_proxy_metrics(text: str) -> dict[str, dict[str, float | int | None]]:
    """Summarize EPD proxy stage histograms (epd_stage_seconds_* metrics)."""
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    buckets: dict[str, list[tuple[float | None, int]]] = {}
    for match in _STAGE_TOTAL_PATTERN.finditer(text):
        kind, stage, raw_value = match.groups()
        if kind == "count":
            counts[stage] = int(float(raw_value))
        else:
            sums[stage] = float(raw_value)
    for match in _STAGE_BUCKET_PATTERN.finditer(text):
        stage, raw_bound, raw_count = match.groups()
        bound = None if raw_bound == "+Inf" else float(raw_bound)
        # Prometheus writes integer counters, but exporters/rewriters may emit 4.0.
        buckets.setdefault(stage, []).append((bound, int(float(raw_count))))
    stages: dict[str, dict[str, float | int | None]] = {}
    for stage in sorted(set(counts) | set(buckets)):
        ordered_buckets = sorted(buckets.get(stage, []), key=lambda item: math.inf if item[0] is None else item[0])
        bucket_total = next((cumulative for bound, cumulative in ordered_buckets if bound is None), None)
        count = counts.get(stage, bucket_total)
        stage_sum = sums.get(stage)
        stages[stage] = {
            "count": count,
            # The exporter may expose only part of the histogram; a missing sum
            # stays unmeasured instead of turning into 0.0.
            "mean_ms": stage_sum / count * MILLISECONDS_PER_SECOND if count and stage_sum is not None else None,
            "p50_ms": _bucket_percentile(ordered_buckets, 0.50),
            "p99_ms": _bucket_percentile(ordered_buckets, 0.99),
        }
    return stages


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metrics_view(data: Any) -> dict[str, Any]:
    """Accept this tool's report, a raw benchmark dict, or vllm result JSON."""
    if not isinstance(data, dict):
        return {}
    benchmark = data.get("benchmark")
    if isinstance(benchmark, dict):
        return benchmark
    mapped: dict[str, Any] = dict(data)
    for alias, canonical in VLLM_RESULT_ALIASES.items():
        if alias in data and canonical not in mapped:
            mapped[canonical] = data[alias]
    return mapped


def compare_with_tolerance(
    current: Any,
    baseline: Any,
    tol: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare current vs baseline metrics; lower latency is better."""
    current_metrics = _metrics_view(current)
    baseline_metrics = _metrics_view(baseline)
    comparisons: list[dict[str, Any]] = []
    regressions: list[str] = []
    improvements: list[str] = []
    for name in COMPARE_METRICS:
        current_value = _numeric(current_metrics.get(name))
        baseline_value = _numeric(baseline_metrics.get(name))
        entry: dict[str, Any] = {
            "metric": name,
            "current": current_value,
            "baseline": baseline_value,
            "delta_ratio": None,
            "status": "missing",
        }
        if current_value is not None and baseline_value not in (None, 0.0):
            delta = (current_value - baseline_value) / baseline_value
            entry["delta_ratio"] = delta
            if name in HIGHER_IS_BETTER:
                regressed = delta < -tol
                improved = delta > tol
            elif name in LOWER_IS_BETTER:
                regressed = delta > tol
                improved = delta < -tol
            else:
                regressed = improved = False
            if regressed:
                entry["status"] = "regressed"
                regressions.append(name)
            elif improved:
                entry["status"] = "improved"
                improvements.append(name)
            else:
                entry["status"] = "within_tolerance"
        comparisons.append(entry)
    return {
        "tolerance": tol,
        "metrics": comparisons,
        "regressions": regressions,
        "improvements": improvements,
        "has_regression": bool(regressions),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _numeric(value)
    return UNMEASURED if number is None else f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    number = _numeric(value)
    return UNMEASURED if number is None else str(int(number))


def _status(value: Any) -> str:
    return UNMEASURED if _numeric(value) is None else MEASURED


def _stage_status(summary: dict[str, Any]) -> str:
    """A stage row counts as measured only when every shown column has a number."""
    measured = all(_numeric(summary.get(key)) is not None for key in ("count", "mean_ms", "p50_ms", "p99_ms"))
    return MEASURED if measured else UNMEASURED


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable report; unmeasured fields say 未采集."""
    benchmark = _metrics_view(report)
    stages = report.get("stages") if isinstance(report.get("stages"), dict) else {}
    comparison = report.get("comparison") if isinstance(report.get("comparison"), dict) else None
    lines = [
        "# EPD benchmark report",
        "",
        "## Environment",
        "",
        f"- Model: {report.get('model') or UNMEASURED}",
        f"- Generated at: {report.get('generated_at') or UNMEASURED}",
    ]
    command = report.get("command")
    if command:
        executed = report.get("benchmark_executed", True)
        label = "Benchmark command" if executed else "Benchmark command (not executed)"
        lines.append(f"- {label}: {' '.join(str(part) for part in command)}")
    else:
        lines.append(f"- Benchmark command: {UNMEASURED}")

    throughput_rows = {
        "Request throughput (req/s)": "request_throughput_rps",
        "Output token throughput (tok/s)": "output_token_throughput",
        "Total token throughput (tok/s)": "total_token_throughput",
        "Mean TTFT (ms)": "mean_ttft_ms",
        "Mean TPOT (ms)": "mean_tpot_ms",
    }
    lines += [
        "",
        "## Throughput and latency",
        "",
        "| Metric | Value | Status |",
        "| --- | --- | --- |",
    ]
    for label, key in throughput_rows.items():
        lines.append(f"| {label} | {_fmt(benchmark.get(key))} | {_status(benchmark.get(key))} |")
    lines += [
        "",
        "## Stage metrics (encode / prefill / decode)",
        "",
    ]
    if stages:
        lines += [
            "| Stage | Count | Mean (ms) | P50 (ms) | P99 (ms) | Status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for stage in sorted(stages):
            summary = stages[stage]
            lines.append(
                f"| {stage} | {_fmt_int(summary.get('count'))} | {_fmt(summary.get('mean_ms'))} | "
                f"{_fmt(summary.get('p50_ms'))} | {_fmt(summary.get('p99_ms'))} | {_stage_status(summary)} |"
            )
        if report.get("stage_percentile_source") == STAGE_PERCENTILE_BUCKET:
            lines.append("")
            lines.append(
                "P50/P99 are Prometheus histogram bucket upper bounds, not exact quantiles: "
                "the proxy /metrics endpoint does not expose the individual stage durations."
            )
    else:
        lines.append(UNMEASURED)

    ec_bytes = report.get("ec_transfer_bytes")
    hbm_peak = report.get("hbm_peak_gb")
    lines += [
        "",
        "## Transfer and memory",
        "",
        "| Metric | Value | Status |",
        "| --- | --- | --- |",
        f"| EC transfer bytes | {_fmt_int(ec_bytes)} | {_status(ec_bytes)} |",
        f"| HBM peak (GB) | {_fmt(hbm_peak)} | {_status(hbm_peak)} |",
        "",
        "## Baseline comparison",
        "",
    ]
    if comparison:
        lines += [
            "| Metric | Current | Baseline | Delta | Status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for entry in comparison.get("metrics", []):
            delta = entry.get("delta_ratio")
            delta_text = UNMEASURED if delta is None else f"{float(delta) * 100:+.2f}%"
            lines.append(
                f"| {entry.get('metric')} | {_fmt(entry.get('current'))} | {_fmt(entry.get('baseline'))} | "
                f"{delta_text} | {entry.get('status')} |"
            )
        if comparison.get("has_regression"):
            lines.append("")
            lines.append("Regression threshold exceeded: " + ", ".join(comparison.get("regressions", [])))
    else:
        lines.append(UNMEASURED)

    notes = report.get("notes") or []
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    """Render the machine-readable JSON report."""
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def _fetch_text(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname is None:
        raise ValueError(f"metrics endpoint URL {url!r} has no host")
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported metrics endpoint URL {url!r}: only http and https are supported")
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=30)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read().decode("utf-8")
        except http.client.HTTPException as exc:
            # BadStatusLine/IncompleteRead and friends are not OSError subclasses:
            # without this guard a broken ingress page crashes the whole report.
            raise RuntimeError(f"metrics endpoint {url!r} did not return a valid HTTP response: {exc}") from exc
        if response.status >= 400:
            raise RuntimeError(f"metrics endpoint returned HTTP {response.status}")
        return body
    finally:
        connection.close()


def _build_bench_command(args: argparse.Namespace) -> list[str]:
    """Return the vllm bench serve argv the report records."""
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--endpoint",
        args.endpoint,
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.concurrency),
        "--metric-percentiles",
        "50,90,99",
    ]
    if args.base_url:
        command += ["--base-url", args.base_url]
    else:
        command += ["--host", args.host, "--port", str(args.port)]
    if args.dataset:
        command += ["--dataset-name", args.dataset]
    else:
        command += [
            "--dataset-name",
            "random",
            "--input-len",
            str(args.input_len),
            "--output-len",
            str(args.output_len),
        ]
    return command


def _load_stage_metrics(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    """Return the stage summary and how its percentiles were obtained."""
    if args.stage_metrics_json:
        rows = json.loads(Path(args.stage_metrics_json).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("--stage-metrics-json must contain a JSON list of {stage, seconds} rows")
        return aggregate_stage_metrics(rows), STAGE_PERCENTILE_EXACT
    if args.proxy_metrics_url:
        return parse_proxy_metrics(_fetch_text(args.proxy_metrics_url)), STAGE_PERCENTILE_BUCKET
    return {}, None


def _load_ec_bytes(args: argparse.Namespace) -> int | None:
    """Return manually supplied EC bytes or the safetensors file sizes on disk."""
    if args.ec_transfer_bytes is not None:
        return int(args.ec_transfer_bytes)
    if args.ec_shared_storage_path:
        root = Path(args.ec_shared_storage_path)
        if root.is_dir():
            # A broken symlink or a file removed mid-scan must not abort the report.
            try:
                return sum(path.stat().st_size for path in root.glob("**/*.safetensors") if path.is_file())
            except OSError:
                return None
    return None


def _tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    """Return the end of a process log, bounded so the report stays readable."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return "..." + stripped[-limit:]


def _validate_inputs(args: argparse.Namespace) -> str | None:
    """Return a user-facing message when the requested report cannot be trusted."""
    if not 0 < args.tolerance <= 1:
        return f"--tolerance must be in (0, 1], got {args.tolerance}"
    if args.fail_on_regression and not args.baseline_json:
        return "--fail-on-regression needs --baseline-json to compare against"
    if args.bench_output and args.dry_run:
        return "--bench-output (parse a previous run) and --dry-run (do not run) cannot be combined"
    if args.bench_output and not Path(args.bench_output).is_file():
        return f"--bench-output {args.bench_output} does not exist or is not a file"
    if args.baseline_json and not Path(args.baseline_json).is_file():
        return f"--baseline-json {args.baseline_json} does not exist or is not a file"
    for option, path_str in (("--report-md", args.report_md), ("--report-json", args.report_json)):
        parent = Path(path_str).parent
        if not parent.is_dir():
            return f"{option} parent directory {parent} does not exist"
    return None


def _write_reports(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """Write both reports through temporary files so no half-written pair is left."""
    targets = (
        (Path(args.report_md), render_markdown(report)),
        (Path(args.report_json), render_json(report)),
    )
    temporary = [(target.with_name(target.name + ".tmp"), target, text) for target, text in targets]
    try:
        for temp, _, text in temporary:
            temp.write_text(text, encoding="utf-8")
        for temp, target, _ in temporary:
            os.replace(temp, target)
    finally:
        for temp, _, _ in temporary:
            temp.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or parse an EPD benchmark and emit markdown + JSON reports.",
        epilog=(
            "Exit codes: 0 report written; 1 a regression exceeded --tolerance "
            "(with --fail-on-regression); 2 the harness could not produce a trustworthy "
            "report (bad input path, benchmark failure or timeout, missing report directory)."
        ),
    )
    parser.add_argument("--model", required=True, help="Model name used by vllm bench serve and the report")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of benchmark prompts (default: 100)")
    parser.add_argument("--input-len", type=int, default=1024, help="Random-dataset input length (default: 1024)")
    parser.add_argument("--output-len", type=int, default=128, help="Random-dataset output length (default: 128)")
    parser.add_argument("--dataset", default=None, help="Dataset name for vllm bench serve (default: random)")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrency (default: 1)")
    parser.add_argument("--base-url", default=None, help="Existing service base URL (e.g. the EPD proxy)")
    parser.add_argument("--host", default="127.0.0.1", help="Service host when --base-url is not set")
    parser.add_argument("--port", type=int, default=8000, help="Service port when --base-url is not set")
    parser.add_argument("--backend", default="openai-chat", help="vllm bench backend (default: openai-chat)")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="Benchmark endpoint")
    parser.add_argument("--baseline-json", default=None, help="Baseline report/result JSON for comparison")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Regression tolerance (default: 0.05)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when the comparison has a regression",
    )
    parser.add_argument("--report-md", default="epd_report.md", help="Markdown output path")
    parser.add_argument("--report-json", default="epd_report.json", help="JSON output path")
    parser.add_argument("--bench-output", default=None, help="Parse an existing bench stdout file instead of running")
    parser.add_argument("--dry-run", action="store_true", help="Do not run the benchmark; record the command only")
    parser.add_argument(
        "--bench-timeout-seconds",
        type=float,
        default=DEFAULT_BENCH_TIMEOUT_SECONDS,
        help="Kill the benchmark and exit 2 after this many seconds (default: 7200)",
    )
    parser.add_argument("--proxy-metrics-url", default=None, help="EPD proxy /metrics URL for stage metrics")
    parser.add_argument("--stage-metrics-json", default=None, help="JSON list of {stage, seconds} rows")
    parser.add_argument("--ec-shared-storage-path", default=None, help="ExampleConnector shared dir for EC bytes")
    parser.add_argument("--ec-transfer-bytes", type=int, default=None, help="Manually measured EC transfer bytes")
    parser.add_argument("--hbm-peak-gb", type=float, default=None, help="Manually measured HBM peak in GB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_error = _validate_inputs(args)
    if input_error is not None:
        print(f"error: {input_error}", file=sys.stderr)
        return ERROR_EXIT_CODE

    notes: list[str] = []
    command = _build_bench_command(args)
    exit_code = 0

    if args.bench_output:
        text = Path(args.bench_output).read_text(encoding="utf-8")
        notes.append(f"parsed existing benchmark output from {args.bench_output}; no benchmark command was executed")
    elif args.dry_run:
        text = ""
        notes.append("dry-run: the benchmark command was not executed and no numbers were produced")
    else:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.bench_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            text = ""
            notes.append(f"benchmark command timed out after {args.bench_timeout_seconds:g}s and was killed")
            tail = _tail("\n".join(part for part in (exc.stdout, exc.stderr) if isinstance(part, str)))
            if tail:
                notes.append(f"benchmark output before the timeout: {tail}")
            exit_code = ERROR_EXIT_CODE
        except OSError as exc:
            text = ""
            notes.append(f"failed to execute benchmark command: {exc}")
            exit_code = ERROR_EXIT_CODE
        else:
            text = completed.stdout + "\n" + completed.stderr
            if completed.returncode != 0:
                notes.append(f"benchmark command failed with exit code {completed.returncode}")
                tail = _tail(text)
                if tail:
                    notes.append(f"benchmark output: {tail}")
                exit_code = ERROR_EXIT_CODE

    benchmark = parse_bench_output(text)
    stages: dict[str, Any] = {}
    stage_percentile_source: str | None = None
    stages_collected = True
    try:
        stages, stage_percentile_source = _load_stage_metrics(args)
    except (OSError, RuntimeError, ValueError) as exc:
        stages_collected = False
        notes.append(f"stage metrics were not collected: {exc}")
    if stages_collected and not stages:
        if args.proxy_metrics_url:
            notes.append(f"no epd_stage_seconds_* metrics found at {args.proxy_metrics_url}")
        if args.stage_metrics_json:
            notes.append(f"no usable {{stage, seconds}} rows found in {args.stage_metrics_json}")

    ec_transfer_bytes = _load_ec_bytes(args)
    if ec_transfer_bytes is None:
        notes.append("EC transfer bytes were not collected")
        if args.ec_shared_storage_path and not Path(args.ec_shared_storage_path).is_dir():
            notes.append(f"EC shared storage path {args.ec_shared_storage_path} is not a directory")
    elif args.ec_transfer_bytes is None:
        notes.append(
            "EC transfer bytes are the sum of the *.safetensors file sizes under "
            f"{args.ec_shared_storage_path} at report time, not bytes transferred by this run"
        )
    if args.hbm_peak_gb is None:
        notes.append("HBM peak was not collected")

    report: dict[str, Any] = {
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # --bench-output parses a previous run, so the command in this report was
        # not executed; --dry-run does not run it either but still records it.
        "command": None if args.bench_output else command,
        "benchmark_executed": not (args.dry_run or args.bench_output),
        "benchmark": benchmark,
        "stages": stages,
        "stage_percentile_source": stage_percentile_source,
        "ec_transfer_bytes": ec_transfer_bytes,
        "hbm_peak_gb": args.hbm_peak_gb,
        "comparison": None,
        "notes": notes,
    }
    if args.baseline_json:
        try:
            baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # A broken baseline must not throw away the freshly measured numbers.
            notes.append(f"baseline {args.baseline_json} was not usable, comparison skipped: {exc}")
        else:
            report["comparison"] = compare_with_tolerance(report, baseline, args.tolerance)

    _write_reports(args, report)
    print(f"wrote {args.report_md} and {args.report_json}")

    if exit_code != 0:
        return exit_code
    comparison = report.get("comparison")
    if args.fail_on_regression and isinstance(comparison, dict) and comparison.get("has_regression"):
        return REGRESSION_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
