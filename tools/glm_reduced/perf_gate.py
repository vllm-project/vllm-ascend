# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation and comparison for the GLM5.x 11-layer performance gate.

The gate compares two single-round offline runs of the SAME checkpoint: the
committed baseline and a fresh candidate with an identical hardware tag, engine
settings, workload shapes and iteration counts. It fails closed: a missing or
malformed record, an identity mismatch, or an absent threshold is a hard
failure, never a skip. Thresholds are supplied per hardware by the caller; this
module deliberately carries no default tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FORMAT = "glm5x-perf-result-v1"
DATA = Path(__file__).parent / "data" / "glm52" / "perf"

# Metrics compared, and the three statistics every record must carry.
METRICS = ("output_tokens_per_second", "ttft_seconds", "tpot_seconds", "seconds")
STATISTICS = ("median", "min", "max")
THROUGHPUT_METRIC = "output_tokens_per_second"
LATENCY_METRICS = ("ttft_seconds", "tpot_seconds", "seconds")

# Identity fields that must match exactly between baseline and candidate.
IDENTITY_KEYS = (
    "model",
    "hardware",
    "engine_settings",
    "workloads",
    "warmup_iterations",
    "measured_iterations",
)


def sha256(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_engine(settings: dict) -> dict:
    """Engine settings minus the checkpoint path, which legitimately differs per run."""
    return {key: value for key, value in settings.items() if key != "model"}


def workload_id(run: dict) -> str:
    return str(run.get("input_tokens")) + "x" + str(run.get("output_tokens")) + "x" + str(run.get("concurrency"))


def _is_positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate(payload: dict, where: str = "perf result") -> list[str]:
    """Structural validity beyond JSON parsing; returns human-readable problems."""
    problems: list[str] = []
    if payload.get("format") != FORMAT:
        problems.append(where + ": not a " + FORMAT + " document")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        problems.append(where + ": identity must be an object")
    else:
        for key in IDENTITY_KEYS:
            if key not in identity:
                problems.append(where + ": identity is missing " + key)
        if not isinstance(identity.get("engine_settings"), dict) or not identity.get("engine_settings"):
            problems.append(where + ": engine_settings must be a non-empty object")
        workloads = identity.get("workloads")
        invalid_workloads = (
            not isinstance(workloads, list)
            or not workloads
            or any(
                not isinstance(item, list)
                or len(item) != 3
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in item)
                for item in workloads
            )
        )
        if invalid_workloads:
            problems.append(where + ": workloads must be a non-empty list of positive input/output/concurrency triples")
        for key in ("warmup_iterations", "measured_iterations"):
            value = identity.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(where + ": identity " + key + " must be a positive integer")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        problems.append(where + ": runs must be a non-empty list")
        return problems
    seen = set()
    for run in runs:
        if not isinstance(run, dict):
            problems.append(where + ": every run must be an object")
            continue
        identifier = workload_id(run)
        if identifier in seen:
            problems.append(where + ": duplicate workload " + identifier)
        seen.add(identifier)
        for key in ("input_tokens", "output_tokens", "concurrency"):
            value = run.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                problems.append(where + ": run " + identifier + " has invalid " + key + "=" + repr(value))
        aggregate = run.get("aggregate")
        if not isinstance(aggregate, dict):
            problems.append(where + ": run " + identifier + " has no aggregate object")
            continue
        for metric in METRICS:
            block = aggregate.get(metric)
            if not isinstance(block, dict):
                problems.append(where + ": run " + identifier + " aggregate is missing " + metric)
                continue
            for statistic in STATISTICS:
                value = block.get(statistic)
                if not _is_positive_number(value):
                    problems.append(
                        where
                        + ": run "
                        + identifier
                        + " "
                        + metric
                        + "."
                        + statistic
                        + " is not a positive finite number: "
                        + repr(value)
                    )
    return problems


def load(path: Path | str) -> dict:
    payload = read_json(path)
    problems = validate(payload, str(path))
    if problems:
        raise ValueError("; ".join(problems))
    return payload


def load_registered(path: Path | str, directory: Path = DATA) -> dict:
    """Load a baseline only when it is the pinned fixture recorded in case.json."""
    case = read_json(Path(directory) / "case.json")
    if sha256(path) != case["baseline_sha256"]:
        raise ValueError("Baseline " + Path(path).name + " is not the pinned fixture recorded in case.json")
    return load(path)


def load_baseline(directory: Path = DATA) -> dict:
    case = read_json(Path(directory) / "case.json")
    return load_registered(Path(directory) / case["baseline"], directory)


def identity_problems(baseline: dict, candidate: dict, where: str = "identity") -> list[str]:
    problems: list[str] = []
    left = baseline.get("identity") or {}
    right = candidate.get("identity") or {}
    for key in IDENTITY_KEYS:
        if key == "engine_settings":
            a = canonical_engine(left.get(key) or {})
            b = canonical_engine(right.get(key) or {})
        else:
            a = left.get(key)
            b = right.get(key)
        if a != b:
            problems.append(where + ": " + key + " differs: baseline=" + repr(a) + " candidate=" + repr(b))
    return problems


def compare(
    baseline: dict,
    candidate: dict,
    *,
    min_throughput_fraction: float | None = None,
    max_latency_fraction: float | None = None,
) -> dict:
    """Fail-closed comparison of one candidate record against the pinned baseline."""
    report: dict = {
        "ok": True,
        "problems": [],
        "thresholds": {
            "min_throughput_fraction": min_throughput_fraction,
            "max_latency_fraction": max_latency_fraction,
        },
        "workloads": {},
    }
    if min_throughput_fraction is None and max_latency_fraction is None:
        report["problems"].append(
            "no threshold supplied; a performance comparison without an explicit pass/fail criterion is not a gate"
        )
    for name, value in (
        ("min_throughput_fraction", min_throughput_fraction),
        ("max_latency_fraction", max_latency_fraction),
    ):
        if value is not None and not _is_positive_number(value):
            report["problems"].append("threshold " + name + " must be a positive finite number, got " + repr(value))
    report["problems"].extend(identity_problems(baseline, candidate))
    report["problems"].extend(validate(baseline, "baseline"))
    report["problems"].extend(validate(candidate, "candidate"))
    if report["problems"]:
        report["ok"] = False
        return report

    by_id = {workload_id(run): run for run in baseline["runs"]}
    for run in candidate["runs"]:
        identifier = workload_id(run)
        reference = by_id.get(identifier)
        if reference is None:
            report["problems"].append("candidate workload " + identifier + " is absent from the baseline")
            continue
        entry: dict = {}
        for metric in METRICS:
            base_median = reference["aggregate"][metric]["median"]
            value = run["aggregate"][metric]["median"]
            entry[metric] = {
                "baseline_median": base_median,
                "candidate_median": value,
                "change_pct": 100.0 * (value - base_median) / base_median,
            }
        if min_throughput_fraction is not None:
            fraction = (
                run["aggregate"][THROUGHPUT_METRIC]["median"] / reference["aggregate"][THROUGHPUT_METRIC]["median"]
            )
            entry["throughput_fraction"] = fraction
            if fraction < min_throughput_fraction:
                report["problems"].append(
                    "workload "
                    + identifier
                    + ": output throughput is "
                    + format(fraction, ".4f")
                    + " of the baseline (< "
                    + repr(min_throughput_fraction)
                    + ")"
                )
        if max_latency_fraction is not None:
            for metric in LATENCY_METRICS:
                fraction = run["aggregate"][metric]["median"] / reference["aggregate"][metric]["median"]
                entry[metric + "_fraction"] = fraction
                if fraction > max_latency_fraction:
                    report["problems"].append(
                        "workload "
                        + identifier
                        + ": "
                        + metric
                        + " is "
                        + format(fraction, ".4f")
                        + " of the baseline (> "
                        + repr(max_latency_fraction)
                        + ")"
                    )
        report["workloads"][identifier] = entry
    report["ok"] = not report["problems"]
    return report
