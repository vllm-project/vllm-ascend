#!/usr/bin/env python3
"""Generate a final tuning report from the optimization ledger and result dirs."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOW_CONC_THRESHOLD = 10
LOWER_IS_BETTER = {
    "ttft_avg",
    "ttft_p99",
    "tpot_avg",
    "tpot_p99",
    "latency_avg",
    "latency_p99",
}
FAILURE_VERDICTS = {"STARTUP_FAIL", "BENCHMARK_FAIL", "SKIPPED_CONFLICT"}

METRIC_LABELS = {
    "ttft_p99": "TTFT P99 (ms)",
    "ttft_avg": "TTFT avg (ms)",
    "tpot_p99": "TPOT P99 (ms)",
    "tpot_avg": "TPOT avg (ms)",
    "latency_p99": "E2E Latency P99 (ms)",
    "latency_avg": "E2E Latency avg (ms)",
    "output_token_throughput": "Output tok/s",
}


@dataclass
class IterResult:
    iteration: int
    phase: str
    lever_name: str
    hypothesis: str
    change_desc: str
    verdict: str
    carry_forward: bool
    metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    notes: str = ""
    failure_stage: str = ""
    reason: str = ""
    evidence: str = ""


@dataclass
class TuningRun:
    model_name: str
    target_metric: str
    baseline_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    best_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    best_config: str = ""
    iterations: list[IterResult] = field(default_factory=list)
    winning_levers: list[str] = field(default_factory=list)
    failed_levers: list[str] = field(default_factory=list)
    timestamp: str = ""


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _percentile_row(percentile: list, pct: str) -> dict:
    """Find the row for a given percentile string (e.g. '99%') in evalscope
    benchmark_percentile.json, which is a list of flat dicts."""
    for row in percentile or []:
        if isinstance(row, dict) and str(row.get("Percentiles", "")).strip() == pct:
            return row
    return {}


def _metrics_from_summary(summary: dict, percentile: list) -> dict[str, float]:
    """Build the metric-keyed dict for one concurrency level from a paired
    evalscope v1.8.1 benchmark_summary.json (flat dict) +
    benchmark_percentile.json (list).

    Units: 'Avg Latency (s)' / 'Latency (s)' are seconds -> *1000 to ms;
    'TTFT (ms)' / 'TPOT (ms)' are already ms.
    """
    p99 = _percentile_row(percentile, "99%")
    return {
        "ttft_avg": _safe_float(summary.get("TTFT (ms)")),
        "ttft_p99": _safe_float(p99.get("TTFT (ms)")),
        "tpot_avg": _safe_float(summary.get("TPOT (ms)")),
        "tpot_p99": _safe_float(p99.get("TPOT (ms)")),
        "latency_avg": _safe_float(summary.get("Avg Latency (s)")) * 1000.0,
        "latency_p99": _safe_float(p99.get("Latency (s)")) * 1000.0,
        "output_token_throughput": _safe_float(summary.get("Output Throughput (tok/s)")),
    }


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_json_metrics(results_dir: Path) -> dict[int, dict[str, float]]:
    """Parse evalscope perf v1.8.1 output.

    Layout: <results_dir>/<timestamp>/<name>/parallel_<P>_number_<N>/
            {benchmark_summary.json, benchmark_percentile.json}
    Each parallel_* subdir holds one concurrency level. Returns a map
    concurrency -> metric-keyed dict.
    """
    if not results_dir.exists():
        return {}

    metrics: dict[int, dict[str, float]] = {}
    for sub in sorted(results_dir.rglob("parallel_*_number_*")):
        if not sub.is_dir():
            continue
        summary = _load_json(sub / "benchmark_summary.json")
        percentile = _load_json(sub / "benchmark_percentile.json")
        if not isinstance(summary, dict):
            continue
        if not isinstance(percentile, list):
            percentile = []
        concurrency = int(_safe_float(summary.get("Concurrency"), -1))
        if concurrency < 0:
            continue
        metrics[concurrency] = _metrics_from_summary(summary, percentile)

    if metrics:
        return metrics

    # Fallback: JSONL of flat summary dicts (no paired percentile file).
    for path in sorted(results_dir.rglob("*.jsonl")):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    summary = json.loads(line)
                    if not isinstance(summary, dict):
                        continue
                    concurrency = int(_safe_float(summary.get("Concurrency"), -1))
                    if concurrency < 0:
                        continue
                    metrics[concurrency] = _metrics_from_summary(summary, [])
        except (json.JSONDecodeError, OSError):
            continue
        if metrics:
            return metrics

    return {}


def _parse_number(cell: str) -> float:
    cleaned = re.sub(r"[`*]", "", cell)
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError("no numeric value found")
    return float(match.group(0))


def _metric_key_from_header(header: str) -> str | None:
    normalized = header.strip().lower()
    normalized = normalized.replace("(", " ").replace(")", " ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    mapping = {
        "ttft avg": "ttft_avg",
        "ttft_avg": "ttft_avg",
        "ttft p99": "ttft_p99",
        "ttft_p99": "ttft_p99",
        "tpot avg": "tpot_avg",
        "tpot_avg": "tpot_avg",
        "tpot p99": "tpot_p99",
        "tpot_p99": "tpot_p99",
        "latency avg": "latency_avg",
        "e2e latency avg": "latency_avg",
        "latency_avg": "latency_avg",
        "latency p99": "latency_p99",
        "lat p99": "latency_p99",
        "e2e latency p99": "latency_p99",
        "latency_p99": "latency_p99",
        "tps": "output_token_throughput",
        "tps tok s": "output_token_throughput",
        "output tok s": "output_token_throughput",
        "output token throughput": "output_token_throughput",
    }
    return mapping.get(normalized)


def _parse_metrics_table(block: str) -> dict[int, dict[str, float]]:
    lines = block.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("|") and "Concurrency" in line:
            header_cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if idx + 1 >= len(lines):
                return {}
            metrics: dict[int, dict[str, float]] = {}
            for row in lines[idx + 2 :]:
                if not row.strip().startswith("|"):
                    break
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                if len(cells) != len(header_cells):
                    continue
                conc_match = re.search(r"\d+", cells[0])
                if not conc_match:
                    continue
                concurrency = int(conc_match.group(0))
                row_metrics: dict[str, float] = {}
                for header, cell in zip(header_cells[1:], cells[1:]):
                    metric_key = _metric_key_from_header(header)
                    if not metric_key:
                        continue
                    try:
                        row_metrics[metric_key] = _parse_number(cell)
                    except ValueError:
                        continue
                if row_metrics:
                    metrics[concurrency] = row_metrics
            return metrics
    return {}


def _extract_field(block: str, field_name: str) -> str:
    pattern = re.compile(rf"\*\*{re.escape(field_name)}\*\*:\s*(.+)")
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def _extract_evidence(block: str) -> str:
    evidence_block = re.search(
        r"\*\*Evidence\*\*:\s*(.*?)(?=\n\*\*|\Z)",
        block,
        flags=re.DOTALL,
    )
    if not evidence_block:
        return ""
    lines = []
    for line in evidence_block.group(1).splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            lines.append(line)
    return " | ".join(lines)


def _parse_ledger(ledger_path: Path) -> list[IterResult]:
    if not ledger_path.exists():
        return []

    text = ledger_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^## Iteration (\d+)\s+[—-]\s+(.+?)\n(.*?)(?=^## Iteration \d+\s+[—-]\s+|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    results: list[IterResult] = []
    for match in pattern.finditer(text):
        iteration = int(match.group(1))
        title = match.group(2).strip()
        block = match.group(3)
        verdict = _extract_field(block, "Verdict") or "NEUTRAL"
        carry_forward = verdict == "WIN"
        carry_field = _extract_field(block, "Carry forward")
        if carry_field:
            carry_forward = carry_field.upper() == "YES"

        results.append(
            IterResult(
                iteration=iteration,
                phase=title,
                lever_name=title,
                hypothesis=_extract_field(block, "Hypothesis"),
                change_desc=_extract_field(block, "Change").strip("`"),
                verdict=verdict,
                carry_forward=carry_forward,
                metrics=_parse_metrics_table(block),
                notes=_extract_field(block, "Notes"),
                failure_stage=_extract_field(block, "Failure stage"),
                reason=_extract_field(block, "Reason"),
                evidence=_extract_evidence(block),
            )
        )

    return results


def _improvement_pct(baseline: float, current: float, metric: str) -> float | None:
    if baseline == 0:
        return None
    if metric in LOWER_IS_BETTER:
        return (baseline - current) / baseline * 100
    return (current - baseline) / baseline * 100


def _fmt_improvement(baseline: float, current: float, metric: str) -> str:
    improvement = _improvement_pct(baseline, current, metric)
    if improvement is None:
        return "N/A"
    sign = "+" if improvement >= 0 else ""
    return f"{sign}{improvement:.1f}%"


def render_report(run: TuningRun) -> str:
    lines: list[str] = []
    append = lines.append

    verdict_counts: dict[str, int] = {}
    for iteration in run.iterations:
        verdict_counts[iteration.verdict] = verdict_counts.get(iteration.verdict, 0) + 1

    append(f"# vLLM-Ascend Tuning Report — {run.model_name}")
    append("")
    append(f"> Generated: {run.timestamp}")
    append("")

    append("## Summary")
    append("")
    append("| Item | Value |")
    append("|------|-------|")
    append(f"| Model | `{run.model_name}` |")
    append(f"| Target Metric | `{METRIC_LABELS.get(run.target_metric, run.target_metric)}` |")
    append(f"| Total Iterations | `{len(run.iterations)}` |")
    for verdict in ("WIN", "NEUTRAL", "LOSS", "STARTUP_FAIL", "BENCHMARK_FAIL", "SKIPPED_CONFLICT"):
        append(f"| {verdict} | `{verdict_counts.get(verdict, 0)}` |")

    baseline_c1 = run.baseline_metrics.get(1, {}).get(run.target_metric, 0.0)
    best_c1 = run.best_metrics.get(1, {}).get(run.target_metric, 0.0)
    append(
        f"| Overall improvement (c=1, {METRIC_LABELS.get(run.target_metric, run.target_metric)}) "
        f"| `{_fmt_improvement(baseline_c1, best_c1, run.target_metric)}` |"
    )
    append("")

    append("## Baseline vs Best Configuration")
    append("")
    conc_levels = sorted(
        {
            *[c for c in run.baseline_metrics if c < LOW_CONC_THRESHOLD],
            *[c for c in run.best_metrics if c < LOW_CONC_THRESHOLD],
        }
    ) or [1, 4, 8]
    metrics = [run.target_metric, "ttft_avg", "tpot_avg", "latency_avg", "output_token_throughput"]
    deduped_metrics = list(dict.fromkeys(metrics))
    append("| Concurrency | Metric | Baseline | Best | Improvement |")
    append("|-------------|--------|----------|------|-------------|")
    for concurrency in conc_levels:
        for metric in deduped_metrics:
            baseline = run.baseline_metrics.get(concurrency, {}).get(metric, 0.0)
            best = run.best_metrics.get(concurrency, {}).get(metric, 0.0)
            marker = " ★" if concurrency == 1 and metric == run.target_metric else ""
            append(
                f"| `{concurrency}`{marker} | {METRIC_LABELS.get(metric, metric)} "
                f"| {baseline:.1f} | {best:.1f} | {_fmt_improvement(baseline, best, metric)} |"
            )
    append("")

    append("## Winning Configuration")
    append("")
    if run.best_config:
        append("```bash")
        append(run.best_config)
        append("```")
    else:
        append("> No winning lever beat the baseline. Keep the baseline configuration.")
    append("")

    append("## Optimization History")
    append("")
    append("| # | Phase / Lever | Verdict | c=1 target improvement | Notes |")
    append("|---|---------------|---------|-----------------------|-------|")
    for iteration in run.iterations:
        current = iteration.metrics.get(1, {}).get(run.target_metric, 0.0)
        improvement = _fmt_improvement(baseline_c1, current, run.target_metric) if current else "—"
        note = iteration.reason or iteration.notes or "—"
        append(
            f"| {iteration.iteration} | {iteration.lever_name} | `{iteration.verdict}` | {improvement} | {note[:80]} |"
        )
    append("")

    non_wins = [item for item in run.iterations if item.verdict in {"LOSS", "NEUTRAL"}]
    if non_wins:
        append("## Levers That Did Not Help")
        append("")
        for item in non_wins:
            append(
                f"- **{item.lever_name}** (`{item.verdict}`): "
                f"{item.reason or item.notes or item.hypothesis or 'no extra note recorded'}"
            )
        append("")

    failed = [item for item in run.iterations if item.verdict in FAILURE_VERDICTS]
    if failed:
        append("## Failed / Incompatible Levers")
        append("")
        append("| Lever | Verdict | Stage | Reason | Evidence |")
        append("|-------|---------|-------|--------|----------|")
        for item in failed:
            append(
                f"| `{item.lever_name}` | `{item.verdict}` | `{item.failure_stage or 'unknown'}` "
                f"| {item.reason or '—'} | {item.evidence or '—'} |"
            )
        append("")

    append("## Recommended Next Steps")
    append("")
    if not run.winning_levers:
        append("- No tuning win was recorded. Validate bottlenecks with `msprof` or a newer CANN / vllm-ascend build.")
        append("- Review failed levers first; environment issues are often easier to unlock than new kernel work.")
    else:
        append("- Re-run the best configuration with a broader concurrency sweep before publishing final numbers.")
        append("- Promote stable winning flags into a checked config template once the run is repeatable.")
    append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vllm-ascend tuning final report")
    parser.add_argument("--work-dir", type=Path, required=True, help="Tuning run work directory")
    parser.add_argument("--model-name", default="unknown", help="Model name")
    parser.add_argument(
        "--target-metric",
        default="ttft_avg",
        choices=list(METRIC_LABELS.keys()),
        help="Primary optimization target",
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    if not work_dir.exists():
        print(f"ERROR: work-dir {work_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    baseline_metrics = _load_json_metrics(work_dir / "baseline")
    iterations = _parse_ledger(work_dir / "ledger.md")

    winning_levers: list[str] = []
    failed_levers: list[str] = []
    best_metrics = dict(baseline_metrics)
    best_config_lines: list[str] = []

    for iteration in iterations:
        iter_dir = work_dir / f"iter_{iteration.iteration:02d}"
        dir_metrics = _load_json_metrics(iter_dir)
        if dir_metrics:
            iteration.metrics = dir_metrics

        if iteration.carry_forward:
            if iteration.metrics:
                best_metrics = iteration.metrics
            if iteration.change_desc:
                best_config_lines.append(iteration.change_desc)
            winning_levers.append(iteration.lever_name)
        elif iteration.verdict in FAILURE_VERDICTS:
            failed_levers.append(iteration.lever_name)

    run = TuningRun(
        model_name=args.model_name,
        target_metric=args.target_metric,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        best_config="\n".join(best_config_lines),
        iterations=iterations,
        winning_levers=winning_levers,
        failed_levers=failed_levers,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    output_path = work_dir / "final_report.md"
    output_path.write_text(render_report(run), encoding="utf-8")
    print(f"Final report written to: {output_path}")


if __name__ == "__main__":
    main()
