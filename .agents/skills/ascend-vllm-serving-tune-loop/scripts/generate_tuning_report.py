#!/usr/bin/env python3
"""Generate a final tuning report from the optimization ledger and per-iteration results.

Usage:
    python generate_tuning_report.py \\
        --work-dir ./tuning_run/20250611_120000/ \\
        --model-name Qwen3-32B \\
        --target-metric ttft_avg
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


LOW_CONC_THRESHOLD = 10

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
    verdict: str  # WIN / LOSS / NEUTRAL
    carry_forward: bool
    metrics: dict  # {concurrency: {metric: value}}
    notes: str = ""


@dataclass
class TuningRun:
    model_name: str
    target_metric: str
    baseline_metrics: dict = field(default_factory=dict)
    best_metrics: dict = field(default_factory=dict)
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


def _pct_delta(baseline: float, current: float) -> str:
    if baseline == 0:
        return "N/A"
    delta = current - baseline
    pct = delta / baseline * 100
    sign = "+" if pct >= 0 else ""
    bold_open = "**" if abs(pct) >= 1.0 else ""
    bold_close = "**" if abs(pct) >= 1.0 else ""
    return f"{bold_open}{sign}{pct:.1f}%{bold_close}"


def _load_jsonl_metrics(results_dir: Path) -> dict:
    """Load per-concurrency metrics from evalscope JSONL output."""
    metrics: dict = {}
    for pattern in ("*.jsonl", "results.jsonl"):
        for p in results_dir.glob(pattern):
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        c = int(rec.get("concurrency", rec.get("parallel", 0)))
                        if c == 0:
                            continue
                        stats = rec.get("stats", rec)

                        def ms(key: str) -> float:
                            v = stats.get(key, {})
                            if isinstance(v, dict):
                                return _safe_float(v.get("mean", v.get("avg", 0))) * 1000
                            return _safe_float(v) * 1000

                        def ms_p(key: str, pct: str) -> float:
                            v = stats.get(key, {})
                            if isinstance(v, dict):
                                return _safe_float(v.get(pct, 0)) * 1000
                            return 0.0

                        metrics[c] = {
                            "ttft_avg": ms("ttft"),
                            "ttft_p99": ms_p("ttft", "p99"),
                            "tpot_avg": ms("tpot"),
                            "tpot_p99": ms_p("tpot", "p99"),
                            "latency_avg": ms("latency"),
                            "latency_p99": ms_p("latency", "p99"),
                            "output_token_throughput": _safe_float(
                                stats.get("output_token_throughput", stats.get("token_throughput", 0))
                            ),
                        }
            except (json.JSONDecodeError, OSError):
                continue
    return metrics


def _parse_ledger(ledger_path: Path) -> list[IterResult]:
    """Parse ledger.md into IterResult list (best-effort markdown parse)."""
    if not ledger_path.exists():
        return []

    results: list[IterResult] = []
    text = ledger_path.read_text()

    # Split on ## Iteration headers
    blocks = re.split(r"^## Iteration (\d+)", text, flags=re.MULTILINE)
    # blocks: [preamble, iter_num, block_content, iter_num, block_content, ...]
    i = 1
    while i < len(blocks) - 1:
        iter_num = int(blocks[i])
        content = blocks[i + 1]

        phase_match = re.search(r"## Iteration \d+ — (.+)", f"## Iteration {iter_num}{content}")
        phase_lever = phase_match.group(1).strip() if phase_match else "unknown"

        hypothesis = ""
        h_match = re.search(r"\*\*Hypothesis\*\*:\s*(.+)", content)
        if h_match:
            hypothesis = h_match.group(1).strip()

        change_desc = ""
        c_match = re.search(r"\*\*Change\*\*:\s*`(.+?)`", content)
        if c_match:
            change_desc = c_match.group(1).strip()

        verdict = "NEUTRAL"
        v_match = re.search(r"\*\*Verdict\*\*:\s*(WIN|LOSS|NEUTRAL)", content)
        if v_match:
            verdict = v_match.group(1)

        carry = verdict == "WIN"
        cf_match = re.search(r"\*\*Carry forward\*\*:\s*(YES|NO)", content, re.IGNORECASE)
        if cf_match:
            carry = cf_match.group(1).upper() == "YES"

        notes = ""
        n_match = re.search(r"\*\*Notes\*\*:\s*(.+)", content)
        if n_match:
            notes = n_match.group(1).strip()

        results.append(IterResult(
            iteration=iter_num,
            phase=phase_lever,
            lever_name=phase_lever,
            hypothesis=hypothesis,
            change_desc=change_desc,
            verdict=verdict,
            carry_forward=carry,
            metrics={},
            notes=notes,
        ))
        i += 2

    return results


def render_report(run: TuningRun) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# vLLM-Ascend Tuning Report — {run.model_name}")
    a(f"")
    a(f"> Generated: {run.timestamp}")
    a(f"")

    wins = [r for r in run.iterations if r.verdict == "WIN"]
    losses = [r for r in run.iterations if r.verdict == "LOSS"]
    neutrals = [r for r in run.iterations if r.verdict == "NEUTRAL"]

    # ── Executive Summary ────────────────────────────────────────────────────
    a("## Executive Summary")
    a("")
    a(f"| Item | Value |")
    a(f"|------|-------|")
    a(f"| Model | `{run.model_name}` |")
    a(f"| Target Metric | `{METRIC_LABELS.get(run.target_metric, run.target_metric)}` |")
    a(f"| Total Iterations | `{len(run.iterations)}` |")
    a(f"| Winning Levers | `{len(wins)}` |")
    a(f"| Neutral | `{len(neutrals)}` |")
    a(f"| Regression (LOSS) | `{len(losses)}` |")

    # Compute overall improvement at c=1 for target metric
    if run.baseline_metrics and run.best_metrics:
        base_c1 = run.baseline_metrics.get(1, {}).get(run.target_metric, 0)
        best_c1 = run.best_metrics.get(1, {}).get(run.target_metric, 0)
        if base_c1 > 0:
            overall_pct = (best_c1 - base_c1) / base_c1 * 100
            sign = "+" if overall_pct >= 0 else ""
            a(f"| Overall improvement (c=1, {METRIC_LABELS.get(run.target_metric, run.target_metric)}) | `{sign}{overall_pct:.1f}%` |")

    a("")

    # ── Baseline vs Best ─────────────────────────────────────────────────────
    a("## Baseline vs Best Configuration")
    a("")
    a("> Focus: low-concurrency (c < 10), especially c = 1")
    a("")

    primary_metrics = [run.target_metric, "ttft_avg", "tpot_p99", "latency_p99", "output_token_throughput"]
    seen_metrics: list[str] = []
    for m in primary_metrics:
        if m not in seen_metrics:
            seen_metrics.append(m)

    conc_levels = sorted([c for c in (run.baseline_metrics or run.best_metrics or {}).keys() if c < LOW_CONC_THRESHOLD])
    if not conc_levels:
        conc_levels = [1, 2, 4, 8]

    if run.baseline_metrics or run.best_metrics:
        a("| Concurrency | Metric | Baseline | Best | Delta % |")
        a("|-------------|--------|----------|------|---------|")
        for c in conc_levels:
            for metric in seen_metrics:
                base_val = run.baseline_metrics.get(c, {}).get(metric, 0.0)
                best_val = run.best_metrics.get(c, {}).get(metric, 0.0)
                label = METRIC_LABELS.get(metric, metric)
                row_marker = " ⭐" if c == 1 and metric == run.target_metric else ""
                a(
                    f"| `c={c}`{row_marker} | {label} "
                    f"| {base_val:.1f} | {best_val:.1f} | {_pct_delta(base_val, best_val)} |"
                )
        a("")
        a("> ⭐ = primary optimization target")
        a("")
    else:
        a("> No parsed metrics available. Populate `baseline_metrics` and `best_metrics` in TuningRun.")
        a("")

    # ── Winning Configuration ─────────────────────────────────────────────────
    a("## Winning Configuration")
    a("")
    if run.best_config:
        a("Copy-paste to reproduce the best result:")
        a("")
        a("```bash")
        a(run.best_config)
        a("```")
    elif wins:
        a("Winning levers (apply all cumulatively):")
        a("")
        for w in wins:
            a(f"- **{w.lever_name}**: `{w.change_desc}`")
    else:
        a("> No winning levers found. Baseline is the best configuration.")
    a("")

    # ── Optimization History ─────────────────────────────────────────────────
    a("## Optimization History")
    a("")
    if run.iterations:
        a("| # | Phase / Lever | Verdict | c=1 target delta | Notes |")
        a("|---|--------------|---------|-----------------|-------|")
        for r in run.iterations:
            c1_baseline = run.baseline_metrics.get(1, {}).get(run.target_metric, 0.0)
            c1_iter = r.metrics.get(1, {}).get(run.target_metric, 0.0)
            delta_str = _pct_delta(c1_baseline, c1_iter) if c1_iter else "—"
            verdict_icon = {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "NEUTRAL": "➖ NEUTRAL"}.get(r.verdict, r.verdict)
            a(f"| {r.iteration} | {r.lever_name} | {verdict_icon} | {delta_str} | {r.notes[:60] if r.notes else '—'} |")
        a("")
    else:
        a("> No ledger entries found.")
        a("")

    # ── Failed Levers ─────────────────────────────────────────────────────────
    if losses or neutrals:
        a("## Levers That Did Not Help")
        a("")
        for r in losses + neutrals:
            icon = "❌" if r.verdict == "LOSS" else "➖"
            a(f"- {icon} **{r.lever_name}** (`{r.change_desc}`): {r.hypothesis or 'no hypothesis recorded'}")
        a("")

    # ── Recommendations ───────────────────────────────────────────────────────
    a("## Recommended Next Steps")
    a("")
    if len(wins) == 0:
        a("- No tuning wins found. Consider profiling with `msprof` to identify the actual bottleneck:")
        a("  ```bash")
        a("  msprof --application=<server_cmd> --output=./profile_output")
        a("  ```")
        a("- Check if ACLGraph warmup is completing before measurement starts")
        a("- Verify CANN version matches the vllm-ascend release notes")
    elif len(losses + neutrals) > len(wins) * 2:
        a("- Most levers did not help — the bottleneck may be at the hardware/interconnect level")
        a("- Consider profiling decode kernels with `msnpuprofiler` to check NPU utilization")
    else:
        a("- Run a broader concurrency sweep (c = 16, 32, 64) with the winning configuration")
        a("- Consider quantization (W8A8) if further memory-bandwidth reduction is needed")
        a(f"- Document winning configuration in `configs/<model>.yaml` for reproducibility")
    a("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vllm-ascend tuning final report")
    parser.add_argument("--work-dir", type=Path, required=True, help="Tuning run work directory")
    parser.add_argument("--model-name", default="unknown", help="Model name")
    parser.add_argument("--target-metric", default="ttft_avg",
                        choices=list(METRIC_LABELS.keys()), help="Primary optimization target")
    args = parser.parse_args()

    work_dir = args.work_dir
    if not work_dir.exists():
        print(f"ERROR: work-dir {work_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load baseline metrics
    baseline_dir = work_dir / "baseline"
    baseline_metrics = _load_jsonl_metrics(baseline_dir) if baseline_dir.exists() else {}

    # Find best iteration metrics (last WIN or final iteration)
    iterations = _parse_ledger(work_dir / "ledger.md")
    best_metrics = dict(baseline_metrics)
    best_config_lines: list[str] = []
    for r in iterations:
        if r.verdict == "WIN":
            iter_dir = work_dir / f"iter_{r.iteration:02d}"
            iter_metrics = _load_jsonl_metrics(iter_dir) if iter_dir.exists() else {}
            if iter_metrics:
                best_metrics = iter_metrics
            if r.change_desc:
                best_config_lines.append(r.change_desc)

    run = TuningRun(
        model_name=args.model_name,
        target_metric=args.target_metric,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        best_config="\n".join(best_config_lines),
        iterations=iterations,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    report = render_report(run)
    output_path = work_dir / "final_report.md"
    output_path.write_text(report)
    print(f"Final report written to: {output_path}")


if __name__ == "__main__":
    main()
