# SPDX-License-Identifier: Apache-2.0
"""Performance sample aggregation and baseline/candidate comparison.

Pure logic for the performance gate. ``run_perf.py`` (vLLM/NPU boundary)
collects raw per-iteration samples into a JSON document; this module validates
and aggregates them, and compares a candidate against a baseline.

The gate compares two runs of the **same checkpoint** (identical
``checkpoint_id``) — typically the same reduced checkpoint across runtime
revisions or modes. A full-vs-reduced latency ratio measures the crop, not a
regression, and is rejected as a gate input only insofar as the checkpoint ids
differ. The gate fails closed:

- a missing, malformed or mismatched baseline is a hard failure;
- at least one regression threshold must be supplied by the caller
  (per profile/hardware); a comparison without thresholds fails;
- every measured iteration must have produced exactly the declared workload
  token count (fixed work), and all values must be finite and positive.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .compare import check_identity

PERF_FORMAT = "glm-reduced-perf-v2"
REQUIRED_META_KEYS = (
    "model",
    "checkpoint_id",
    "run_id",
    "hardware",
    "workload_id",
    "warmup_iterations",
    "measured_iterations",
    "expected_iteration_tokens",
    "runtime",
    "engine",
)


@dataclass
class PerfRun:
    meta: dict
    latencies_s: list[float]  # per measured iteration, post-warmup
    total_tokens: list[int]  # tokens generated per measured iteration


def validate_perf_run(run: PerfRun, where: str = "perf record") -> list[str]:
    """Structural validity beyond parsing: positive counts, nonempty identity."""
    problems = []
    meta = run.meta
    for key in ("warmup_iterations", "measured_iterations", "expected_iteration_tokens"):
        value = meta.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"{where}: meta {key}={value!r} is not a positive integer")
    if not isinstance(meta.get("runtime"), dict) or not meta["runtime"]:
        problems.append(f"{where}: runtime identity must be a non-empty object")
    if not isinstance(meta.get("engine"), dict):
        problems.append(f"{where}: engine settings must be an object")
    if not run.latencies_s or len(run.latencies_s) != len(run.total_tokens):
        problems.append(f"{where}: empty or mismatched sample lists")
        return problems
    for i, (latency, count) in enumerate(zip(run.latencies_s, run.total_tokens)):
        if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            problems.append(f"{where}: latency sample {i} is not a positive finite number: {latency!r}")
        expected = meta.get("expected_iteration_tokens")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            problems.append(f"{where}: token count sample {i} is not a positive int: {count!r}")
        elif isinstance(expected, int) and expected > 0 and count != expected:
            problems.append(
                f"{where}: iteration {i} produced {count} tokens, expected exactly {expected} "
                "(fixed-workload violation, e.g. early EOS or cache-altered work)"
            )
    return problems


def load_perf(path: str | Path) -> PerfRun:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"perf record {str(path)!r} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PERF_FORMAT:
        raise ValueError(f"{path.name}: not a {PERF_FORMAT} document")
    meta = payload.get("meta") or {}
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{path.name}: meta is missing keys {missing}")
    latencies = payload.get("latencies_s")
    tokens = payload.get("total_tokens")
    if not isinstance(latencies, list) or not isinstance(tokens, list):
        raise ValueError(f"{path.name}: latencies_s/total_tokens must be lists")
    if len(latencies) != meta["measured_iterations"]:
        raise ValueError(
            f"{path.name}: {len(latencies)} latency samples for declared measured_iterations="
            f"{meta['measured_iterations']}"
        )
    run = PerfRun(meta=meta, latencies_s=latencies, total_tokens=tokens)
    problems = validate_perf_run(run, where=path.name)
    if problems:
        raise ValueError("; ".join(problems))
    return PerfRun(meta=meta, latencies_s=[float(v) for v in latencies], total_tokens=list(tokens))


@dataclass
class PerfStats:
    iterations: int
    mean_latency_s: float
    median_latency_s: float
    p90_latency_s: float
    min_latency_s: float
    max_latency_s: float
    mean_throughput_tok_s: float


def aggregate(run: PerfRun) -> PerfStats:
    latencies = sorted(run.latencies_s)
    p90_index = min(len(latencies) - 1, math.ceil(0.9 * len(latencies)) - 1)
    throughputs = [tokens / latency for tokens, latency in zip(run.total_tokens, run.latencies_s)]
    return PerfStats(
        iterations=len(latencies),
        mean_latency_s=statistics.fmean(latencies),
        median_latency_s=statistics.median(latencies),
        p90_latency_s=latencies[p90_index],
        min_latency_s=latencies[0],
        max_latency_s=latencies[-1],
        mean_throughput_tok_s=statistics.fmean(throughputs),
    )


@dataclass
class PerfComparison:
    ok: bool
    problems: list[str] = field(default_factory=list)
    baseline: PerfStats | None = None
    candidate: PerfStats | None = None
    latency_change_pct: float = 0.0
    throughput_change_pct: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "problems": self.problems,
                "baseline": self.baseline.__dict__ if self.baseline else None,
                "candidate": self.candidate.__dict__ if self.candidate else None,
                "latency_change_pct": self.latency_change_pct,
                "throughput_change_pct": self.throughput_change_pct,
            },
            indent=2,
        )


def compare_perf(
    baseline: PerfRun,
    candidate: PerfRun,
    *,
    max_latency_regression_pct: float | None = None,
    min_throughput_change_pct: float | None = None,
) -> PerfComparison:
    report = PerfComparison(ok=True)
    thresholds = {
        "max_latency_regression_pct": max_latency_regression_pct,
        "min_throughput_change_pct": min_throughput_change_pct,
    }
    for name, value in thresholds.items():
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            report.problems.append(f"threshold {name} must be finite, got {value!r}")
    if all(value is None for value in thresholds.values()):
        report.problems.append(
            "no regression threshold supplied; a performance comparison without an explicit pass/fail "
            "criterion is not a gate (supply --max-latency-regression-pct and/or "
            "--min-throughput-change-pct, chosen per profile/hardware)"
        )
    report.problems.extend(
        check_identity(
            baseline.meta,
            candidate.meta,
            ("hardware", "workload_id", "warmup_iterations", "measured_iterations", "expected_iteration_tokens"),
            require_identical_engine=True,
        )
    )
    report.problems.extend(validate_perf_run(baseline, "baseline"))
    report.problems.extend(validate_perf_run(candidate, "candidate"))
    if report.problems:
        report.ok = False
        return report
    report.baseline = aggregate(baseline)
    report.candidate = aggregate(candidate)
    report.latency_change_pct = (
        100.0
        * (report.candidate.median_latency_s - report.baseline.median_latency_s)
        / report.baseline.median_latency_s
    )
    report.throughput_change_pct = (
        100.0
        * (report.candidate.mean_throughput_tok_s - report.baseline.mean_throughput_tok_s)
        / report.baseline.mean_throughput_tok_s
    )
    if max_latency_regression_pct is not None and report.latency_change_pct > max_latency_regression_pct:
        report.problems.append(
            f"median latency regressed {report.latency_change_pct:.2f}% > allowed {max_latency_regression_pct}%"
        )
    if min_throughput_change_pct is not None and report.throughput_change_pct < min_throughput_change_pct:
        report.problems.append(
            f"throughput changed {report.throughput_change_pct:.2f}% < required {min_throughput_change_pct}%"
        )
    report.ok = not report.problems
    return report
