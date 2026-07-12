#!/usr/bin/env python3
"""Generate a Markdown + CSV benchmark report from evalscope perf output.

Usage:
    python generate_report.py \\
        --results-dir outputs/<timestamp>/<model>/ \\
        --output-dir ./benchmark_output/ \\
        --model-name Qwen2.5-7B-Instruct \\
        --config configs/qwen2.5-7b-instruct.yaml \\
        [--npu-info npu_info.txt] \\
        [--vllm-commit abc1234] \\
        [--server-cmd "python -m vllm.entrypoints..."] \\
        [--evalscope-cmd "evalscope perf ..."]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ConcurrencyResult:
    concurrency: int
    total_requests: int
    success_requests: int
    failed_requests: int
    success_rate: float
    # Throughput
    request_throughput_rps: float
    input_token_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    # Latency (ms)
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    # TTFT (ms)
    ttft_avg_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    # TPOT (ms)
    tpot_avg_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p99_ms: float


@dataclass
class BenchmarkReport:
    model_name: str
    model_path: str
    dtype: str
    quantization: Optional[str]
    max_model_len: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    input_tokens: int
    output_tokens: int
    npu_info: str
    cann_version: str
    vllm_ascend_version: str
    vllm_ascend_commit: str
    evalscope_version: str
    server_cmd: str
    evalscope_cmd: str
    timestamp: str
    results: list[ConcurrencyResult] = field(default_factory=list)
    sla: dict = field(default_factory=dict)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_evalscope_summary_txt(summary_path: Path) -> list[ConcurrencyResult]:
    """Parse evalscope performance_summary.txt into ConcurrencyResult list."""
    if not summary_path.exists():
        return []

    results: list[ConcurrencyResult] = []
    text = summary_path.read_text()

    # evalscope outputs per-concurrency blocks; try to parse tables
    # Format varies by version — we parse the JSON result file when available
    return results


def _parse_records(records: list[dict]) -> list[ConcurrencyResult]:
    """Normalize evalscope records into ConcurrencyResult entries."""
    results: list[ConcurrencyResult] = []
    for rec in records:
        stats = rec.get("stats", rec)
        concurrency = int(rec.get("concurrency", rec.get("parallel", 0)))
        if concurrency == 0:
            continue

        def ms(key: str, sub: str = "") -> float:
            base = stats.get(key, stats.get(sub, {}))
            if isinstance(base, dict):
                return _safe_float(base.get("mean", base.get("avg", 0))) * 1000
            return _safe_float(base) * 1000

        def ms_pct(key: str, pct: str) -> float:
            base = stats.get(key, {})
            if isinstance(base, dict):
                return _safe_float(base.get(pct, 0)) * 1000
            return 0.0

        total = int(stats.get("total_requests", stats.get("num_requests", 0)))
        success = int(stats.get("success_requests", stats.get("num_completed", total)))

        results.append(ConcurrencyResult(
            concurrency=concurrency,
            total_requests=total,
            success_requests=success,
            failed_requests=total - success,
            success_rate=_safe_float(stats.get("success_rate", success / max(total, 1))),
            request_throughput_rps=_safe_float(stats.get("request_throughput", stats.get("throughput_rps", 0))),
            input_token_throughput=_safe_float(stats.get("input_token_throughput", 0)),
            output_token_throughput=_safe_float(stats.get("output_token_throughput", stats.get("token_throughput", 0))),
            total_token_throughput=_safe_float(stats.get("total_token_throughput", 0)),
            latency_avg_ms=ms("latency", "e2e_latency"),
            latency_p50_ms=ms_pct("latency", "p50"),
            latency_p90_ms=ms_pct("latency", "p90"),
            latency_p99_ms=ms_pct("latency", "p99"),
            ttft_avg_ms=ms("ttft"),
            ttft_p50_ms=ms_pct("ttft", "p50"),
            ttft_p90_ms=ms_pct("ttft", "p90"),
            ttft_p99_ms=ms_pct("ttft", "p99"),
            tpot_avg_ms=ms("tpot"),
            tpot_p50_ms=ms_pct("tpot", "p50"),
            tpot_p90_ms=ms_pct("tpot", "p90"),
            tpot_p99_ms=ms_pct("tpot", "p99"),
        ))

    results.sort(key=lambda r: r.concurrency)
    return results


def _parse_evalscope_jsonl(jsonl_path: Path) -> list[ConcurrencyResult]:
    """Parse evalscope result JSONL into ConcurrencyResult list."""
    if not jsonl_path.exists():
        return []

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return _parse_records(records)


def _parse_evalscope_json(json_path: Path) -> list[ConcurrencyResult]:
    """Parse evalscope JSON output into ConcurrencyResult list."""
    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    records: list[dict] = []
    if isinstance(data, list):
        records = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        for key in ("results", "records", "data"):
            nested = data.get(key)
            if isinstance(nested, list):
                records = [item for item in nested if isinstance(item, dict)]
                break
        else:
            records = [data]

    return _parse_records(records)


def _parse_evalscope_output_dir(results_dir: Path) -> list[ConcurrencyResult]:
    """Try multiple evalscope output formats."""
    # Try JSONL first
    for pattern in ("*.jsonl", "results.jsonl", "benchmark_results.jsonl"):
        for p in results_dir.glob(pattern):
            results = _parse_evalscope_jsonl(p)
            if results:
                return results
    # Try JSON
    for pattern in ("*.json",):
        for p in results_dir.glob(pattern):
            results = _parse_evalscope_json(p)
            if results:
                return results
    return []


# ── Environment detection ──────────────────────────────────────────────────────

def _get_npu_info() -> str:
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "npu-smi not available"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "npu-smi not available"


def _get_cann_version() -> str:
    version_file = Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg")
    if version_file.exists():
        text = version_file.read_text()
        m = re.search(r"Version\s*=\s*(.+)", text)
        if m:
            return m.group(1).strip()
    try:
        r = subprocess.run(
            ["python3", "-c", "import torch_npu; print(torch_npu.version.cann)"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_vllm_ascend_version() -> str:
    try:
        r = subprocess.run(
            ["python3", "-c", "import vllm_ascend; print(getattr(vllm_ascend, '__version__', 'unknown'))"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _get_evalscope_version() -> str:
    try:
        r = subprocess.run(
            ["python3", "-c", "import evalscope; print(evalscope.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


# ── SLA check ─────────────────────────────────────────────────────────────────

def _check_sla(result: ConcurrencyResult, sla: dict) -> list[str]:
    violations: list[str] = []
    max_ttft = sla.get("max_p99_ttft_ms")
    if max_ttft and result.ttft_p99_ms > max_ttft:
        violations.append(
            f"P99 TTFT {result.ttft_p99_ms:.0f}ms > SLA {max_ttft}ms"
        )
    min_sr = sla.get("min_success_rate")
    if min_sr and result.success_rate < min_sr:
        violations.append(
            f"success rate {result.success_rate:.3f} < SLA {min_sr}"
        )
    min_tput = sla.get("min_output_token_throughput")
    if min_tput and result.output_token_throughput < min_tput:
        violations.append(
            f"output token throughput {result.output_token_throughput:.0f} tok/s < SLA {min_tput} tok/s"
        )
    return violations


# ── Markdown rendering ─────────────────────────────────────────────────────────

def _fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Ascend 910C vLLM Serving Benchmark Report")
    a(f"")
    a(f"> Generated: {report.timestamp}")
    a(f"")

    # ── 1. Environment ────────────────────────────────────────────────────────
    a("## 1. Environment")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| Hardware | Ascend 910C |")
    a(f"| CANN Version | `{report.cann_version}` |")
    a(f"| vllm-ascend Version | `{report.vllm_ascend_version}` |")
    a(f"| vllm-ascend Git Commit | `{report.vllm_ascend_commit}` |")
    a(f"| evalscope Version | `{report.evalscope_version}` |")
    a("")

    if report.npu_info and report.npu_info != "npu-smi not available":
        a("<details><summary>npu-smi info</summary>")
        a("")
        a("```")
        a(report.npu_info)
        a("```")
        a("")
        a("</details>")
        a("")

    # ── 2. Model Configuration ────────────────────────────────────────────────
    a("## 2. Model Configuration")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| Model | `{report.model_name}` |")
    a(f"| Model Path | `{report.model_path}` |")
    a(f"| dtype | `{report.dtype}` |")
    a(f"| Quantization | `{report.quantization or 'None'}` |")
    a(f"| max_model_len | `{report.max_model_len}` |")
    a(f"| Tensor Parallel Size | `{report.tensor_parallel_size}` |")
    a(f"| Pipeline Parallel Size | `{report.pipeline_parallel_size}` |")
    a("")

    # ── 3. Workload Specification ─────────────────────────────────────────────
    a("## 3. Workload Specification")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| Input Tokens | `{report.input_tokens}` |")
    a(f"| Output Tokens | `{report.output_tokens}` |")
    a(f"| Dataset | `random` |")
    a(f"| Stream Mode | `true` (SSE) |")
    a("")

    if not report.results:
        a("> **No benchmark results found.** Check that the results directory contains JSONL output from evalscope.")
        return "\n".join(lines)

    low_conc = [r for r in report.results if r.concurrency < 10]
    single = next((r for r in report.results if r.concurrency == 1), None)

    # ── 4. Low-Concurrency Deep Dive (c < 10) ────────────────────────────────
    a("## 4. Low-Concurrency Analysis (c < 10)")
    a("")
    a("> Low-concurrency results reflect the model's **baseline latency** with minimal queuing.")
    a("> These numbers matter most for latency-sensitive / interactive workloads.")
    a("")

    if single:
        a("### 4.1 Single-Request Baseline (c = 1)")
        a("")
        a("This is the best-case latency the model can achieve — no batching, no queuing pressure.")
        a("")
        a("| Metric | Value |")
        a("|--------|-------|")
        a(f"| End-to-End Latency (avg) | `{_fmt(single.latency_avg_ms, 1)} ms` |")
        a(f"| End-to-End Latency P50  | `{_fmt(single.latency_p50_ms, 1)} ms` |")
        a(f"| End-to-End Latency P99  | `{_fmt(single.latency_p99_ms, 1)} ms` |")
        a(f"| TTFT (avg)              | `{_fmt(single.ttft_avg_ms, 1)} ms` |")
        a(f"| TTFT P50                | `{_fmt(single.ttft_p50_ms, 1)} ms` |")
        a(f"| TTFT P99                | `{_fmt(single.ttft_p99_ms, 1)} ms` |")
        a(f"| TPOT (avg)              | `{_fmt(single.tpot_avg_ms, 1)} ms/token` |")
        a(f"| TPOT P50                | `{_fmt(single.tpot_p50_ms, 1)} ms/token` |")
        a(f"| TPOT P99                | `{_fmt(single.tpot_p99_ms, 1)} ms/token` |")
        a(f"| Output Token Throughput | `{_fmt(single.output_token_throughput, 1)} tok/s` |")
        a(f"| Success Rate            | `{single.success_rate:.3f}` |")
        a("")

    if low_conc:
        a("### 4.2 Low-Concurrency Sweep (c = 1 / 2 / 4 / 8)")
        a("")
        a("| c | Lat avg | Lat P50 | Lat P99 | TTFT avg | TTFT P50 | TTFT P99 | TPOT avg | Output tok/s |")
        a("|---|---------|---------|---------|----------|----------|----------|----------|--------------|")
        for r in low_conc:
            a(
                f"| **{r.concurrency}** "
                f"| {_fmt(r.latency_avg_ms, 1)} | {_fmt(r.latency_p50_ms, 1)} | {_fmt(r.latency_p99_ms, 1)} "
                f"| {_fmt(r.ttft_avg_ms, 1)} | {_fmt(r.ttft_p50_ms, 1)} | {_fmt(r.ttft_p99_ms, 1)} "
                f"| {_fmt(r.tpot_avg_ms, 1)} | {_fmt(r.output_token_throughput, 1)} |"
            )
        a("")

        # Latency inflation analysis
        if single and len(low_conc) > 1:
            a("### 4.3 Latency Inflation vs c = 1")
            a("")
            a("Shows how much latency grows as concurrency increases from the single-request baseline.")
            a("")
            a("| c | Lat P99 delta | TTFT P99 delta | TPOT P99 delta |")
            a("|---|---------------|----------------|----------------|")
            for r in low_conc:
                if r.concurrency == 1:
                    continue
                lat_delta = r.latency_p99_ms - single.latency_p99_ms
                ttft_delta = r.ttft_p99_ms - single.ttft_p99_ms
                tpot_delta = r.tpot_p99_ms - single.tpot_p99_ms
                lat_sign = "+" if lat_delta >= 0 else ""
                ttft_sign = "+" if ttft_delta >= 0 else ""
                tpot_sign = "+" if tpot_delta >= 0 else ""
                a(
                    f"| **{r.concurrency}** "
                    f"| `{lat_sign}{_fmt(lat_delta, 1)} ms` "
                    f"| `{ttft_sign}{_fmt(ttft_delta, 1)} ms` "
                    f"| `{tpot_sign}{_fmt(tpot_delta, 1)} ms` |"
                )
            a("")
    else:
        a("> No results with concurrency < 10 found. Re-run with `--parallel-levels` including 1, 2, 4, 8.")
        a("")

    # ── 5. Full Performance Overview ──────────────────────────────────────────
    a("## 5. Full Performance Overview")
    a("")
    a("| Concurrency | Total Req | Success | Failed | Success Rate | RPS | Input tok/s | Output tok/s |")
    a("|-------------|-----------|---------|--------|--------------|-----|-------------|--------------|")
    for r in report.results:
        sla_flag = ""
        violations = _check_sla(r, report.sla)
        if violations:
            sla_flag = " ⚠️"
        low_marker = " 🔹" if r.concurrency < 10 else ""
        a(
            f"| {r.concurrency}{low_marker} | {r.total_requests} | {r.success_requests} | {r.failed_requests} "
            f"| {r.success_rate:.3f}{sla_flag} | {_fmt(r.request_throughput_rps)} "
            f"| {_fmt(r.input_token_throughput, 0)} | {_fmt(r.output_token_throughput, 0)} |"
        )
    a("")
    a("> 🔹 = low-concurrency (c < 10)  ⚠️ = SLA violation (see Section 7)")
    a("")

    # ── 6. Full Latency Distribution ──────────────────────────────────────────
    a("## 6. Full Latency Distribution (ms)")
    a("")
    a("| Concurrency | Latency avg | Lat P50 | Lat P90 | Lat P99 | TTFT avg | TTFT P50 | TTFT P90 | TTFT P99 | TPOT avg | TPOT P50 | TPOT P99 |")
    a("|-------------|-------------|---------|---------|---------|----------|----------|----------|----------|----------|----------|----------|")
    for r in report.results:
        low_marker = " 🔹" if r.concurrency < 10 else ""
        a(
            f"| {r.concurrency}{low_marker} "
            f"| {_fmt(r.latency_avg_ms, 1)} | {_fmt(r.latency_p50_ms, 1)} | {_fmt(r.latency_p90_ms, 1)} | {_fmt(r.latency_p99_ms, 1)} "
            f"| {_fmt(r.ttft_avg_ms, 1)} | {_fmt(r.ttft_p50_ms, 1)} | {_fmt(r.ttft_p90_ms, 1)} | {_fmt(r.ttft_p99_ms, 1)} "
            f"| {_fmt(r.tpot_avg_ms, 1)} | {_fmt(r.tpot_p50_ms, 1)} | {_fmt(r.tpot_p99_ms, 1)} |"
        )
    a("")

    # ── 7. Best Configuration Summary ────────────────────────────────────────
    a("## 7. Best Configuration Summary")
    a("")
    valid = [r for r in report.results if r.success_rate >= 0.99]
    if valid:
        peak_tput = max(valid, key=lambda r: r.output_token_throughput)
        best_ttft = min(valid, key=lambda r: r.ttft_p99_ms)
        a(f"| Metric | Concurrency | Value |")
        a(f"|--------|-------------|-------|")
        a(f"| Peak Output Token Throughput | `{peak_tput.concurrency}` | `{_fmt(peak_tput.output_token_throughput, 0)} tok/s` |")
        a(f"| Best P99 TTFT | `{best_ttft.concurrency}` | `{_fmt(best_ttft.ttft_p99_ms, 1)} ms` |")
        a(f"| Best P99 Latency | `{min(valid, key=lambda r: r.latency_p99_ms).concurrency}` | `{_fmt(min(valid, key=lambda r: r.latency_p99_ms).latency_p99_ms, 1)} ms` |")
        if single:
            a(f"| Single-Request (c=1) TTFT avg | `1` | `{_fmt(single.ttft_avg_ms, 1)} ms` |")
            a(f"| Single-Request (c=1) TPOT avg | `1` | `{_fmt(single.tpot_avg_ms, 1)} ms/token` |")
        a("")
        a(f"**Recommended concurrency for throughput**: `{peak_tput.concurrency}` (highest output tok/s with ≥99% success rate)")
        if single:
            a(f"**Single-request latency baseline**: TTFT `{_fmt(single.ttft_avg_ms, 1)} ms`, E2E `{_fmt(single.latency_avg_ms, 1)} ms`")
    else:
        a("> No concurrency level achieved ≥99% success rate. Review server logs for errors.")
    a("")

    # SLA violations summary
    all_violations: list[tuple[int, list[str]]] = []
    for r in report.results:
        v = _check_sla(r, report.sla)
        if v:
            all_violations.append((r.concurrency, v))
    if all_violations:
        a("### SLA Violations")
        a("")
        for concurrency, violations in all_violations:
            a(f"**Concurrency {concurrency}**:")
            for v in violations:
                a(f"  - {v}")
        a("")

    # ── 8. Reproducibility ───────────────────────────────────────────────────
    a("## 8. Reproducibility")
    a("")
    a("### Server Launch Command")
    a("```bash")
    a(report.server_cmd or "# Not recorded")
    a("```")
    a("")
    a("### evalscope Benchmark Command")
    a("```bash")
    a(report.evalscope_cmd or "# Not recorded")
    a("```")
    a("")
    a("### Key Ascend Environment Variables")
    a("```bash")
    ascend_env_keys = [
        "ASCEND_RT_VISIBLE_DEVICES", "VLLM_ASCEND_ENABLE_ACLGRAPH",
        "VLLM_ASCEND_ENABLE_NZ", "HCCL_BUFFSIZE",
        "ASCEND_LAUNCH_BLOCKING", "LD_LIBRARY_PATH",
    ]
    for k in ascend_env_keys:
        val = os.environ.get(k)
        if val:
            a(f"export {k}={val}")
    a("```")
    a("")

    return "\n".join(lines)


# ── CSV export ─────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "concurrency", "total_requests", "success_requests", "failed_requests",
    "success_rate", "request_throughput_rps", "input_token_throughput",
    "output_token_throughput", "total_token_throughput",
    "latency_avg_ms", "latency_p50_ms", "latency_p90_ms", "latency_p99_ms",
    "ttft_avg_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "tpot_avg_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms",
]


def write_csv(report: BenchmarkReport, output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in report.results:
            writer.writerow({k: getattr(r, k) for k in CSV_FIELDS})


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report from evalscope output")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing evalscope perf output")
    parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_output"),
                        help="Output directory for report files")
    parser.add_argument("--model-name", default="unknown", help="Model name for report header")
    parser.add_argument("--model-path", default="", help="Model path or ID")
    parser.add_argument("--config", type=Path, help="YAML config file to read model settings from")
    parser.add_argument("--npu-info", type=Path, help="Path to npu-smi output file")
    parser.add_argument("--vllm-commit", default="", help="vllm-ascend git commit hash")
    parser.add_argument("--server-cmd", default="", help="Server launch command for report")
    parser.add_argument("--evalscope-cmd", default="", help="evalscope command for report")
    args = parser.parse_args()

    # Load config if provided
    cfg: dict = {}
    if args.config and yaml:
        try:
            with open(args.config) as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: could not load config {args.config}: {e}", file=sys.stderr)

    model_cfg = cfg.get("model", {})
    server_cfg = cfg.get("server", {})
    workload_cfg = cfg.get("workload", {})
    sla_cfg = cfg.get("sla", {})

    # Parse results
    results = _parse_evalscope_output_dir(args.results_dir)
    if not results:
        print(
            f"Warning: no benchmark results parsed from {args.results_dir}. "
            "The report will be generated with empty tables.",
            file=sys.stderr,
        )

    # NPU info
    if args.npu_info and args.npu_info.exists():
        npu_info = args.npu_info.read_text()
    else:
        npu_info = _get_npu_info()

    report = BenchmarkReport(
        model_name=args.model_name or model_cfg.get("name", "unknown"),
        model_path=args.model_path or model_cfg.get("path", ""),
        dtype=model_cfg.get("dtype", "float16"),
        quantization=model_cfg.get("quantization"),
        max_model_len=int(model_cfg.get("max_model_len", 8192)),
        tensor_parallel_size=int(server_cfg.get("tensor_parallel_size", 1)),
        pipeline_parallel_size=int(server_cfg.get("pipeline_parallel_size", 1)),
        input_tokens=int(workload_cfg.get("input_tokens", 1024)),
        output_tokens=int(workload_cfg.get("output_tokens", 512)),
        npu_info=npu_info,
        cann_version=_get_cann_version(),
        vllm_ascend_version=_get_vllm_ascend_version(),
        vllm_ascend_commit=args.vllm_commit,
        evalscope_version=_get_evalscope_version(),
        server_cmd=args.server_cmd,
        evalscope_cmd=args.evalscope_cmd,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        sla=sla_cfg,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_path = args.output_dir / "benchmark_report.md"
    md_path.write_text(render_markdown(report))
    print(f"Markdown report written to: {md_path}")

    csv_path = args.output_dir / "benchmark_results.csv"
    write_csv(report, csv_path)
    print(f"CSV results written to: {csv_path}")


if __name__ == "__main__":
    main()
