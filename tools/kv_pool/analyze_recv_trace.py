#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize receive traces without treating unknown dependencies as ready."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def quantiles(values):
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return (ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)) / 1e6

    return {"count": len(values), **{f"p{p}_ms": percentile(p / 100) for p in (50, 95, 99)}}


def summarize(trace):
    durations = defaultdict(list)
    per_task = defaultdict(lambda: defaultdict(int))
    starts = {}
    previous_hol = None
    hol_intervals = []
    hol_evidence = []
    unknown_samples = 0
    errors = []
    resource_peaks = {}
    for row in trace["events"]:
        task, name, timestamp = row["task_id"], row["event"], row["ts_ns"]
        if name == "resources":
            for key, value in row.items():
                if key not in ("task_id", "event", "ts_ns") and value is not None:
                    resource_peaks[key] = max(resource_peaks.get(key, 0), value)
        if name == "enqueue":
            starts[(task, "queue")] = timestamp
        elif name == "dequeue":
            begin = starts.pop((task, "queue"), None)
            if begin is not None:
                durations["queue"].append(timestamp - begin)
                per_task["queue"][task] += timestamp - begin
        elif name.endswith("_begin") or name == "backend_submit":
            phase = "backend" if name == "backend_submit" else name.removesuffix("_begin")
            starts[(task, phase)] = timestamp
        elif name.endswith("_end") or name == "backend_complete" or name.endswith("_timeout"):
            phase = "backend" if name == "backend_complete" else name.removesuffix("_end").removesuffix("_timeout")
            begin = starts.pop((task, phase), None)
            if begin is not None:
                durations[phase].append(timestamp - begin)
                per_task[phase][task] += timestamp - begin
        if name in ("error", "observer_error", "probe_error", "backend_error", "metrics_error"):
            errors.append(row)
        if name == "readiness_sample":
            unknown_samples += int(any(value is None for _, deps in row["dependencies"] for value in deps.values()))
            followers = row["ready_followers"]
            if followers:
                # Consecutive samples are discrete observations, NOT proof
                # that readiness was uninterrupted between them.
                hol_evidence.append(row)
                if previous_hol is not None and previous_hol["task_id"] == task:
                    hol_intervals.append(timestamp - previous_hol["ts_ns"])
                previous_hol = row
            else:
                previous_hol = None
        elif name in ("complete", "error", "cancelled", "dequeue") or name.endswith(("_begin", "_end", "_timeout")):
            previous_hol = None
    return {
        "schema_version": trace["schema_version"],
        "dropped": trace["dropped"],
        "stage_distributions": {name: quantiles(list(values.values())) for name, values in sorted(per_task.items())},
        "span_distributions": {name: quantiles(values) for name, values in sorted(durations.items())},
        "hol_positive_observations": len(hol_evidence),
        "hol_head_tasks": len({row["task_id"] for row in hol_evidence}),
        "hol_consecutive_sample_intervals": quantiles(hol_intervals),
        "hol_observed_span_ms": sum(hol_intervals) / 1e6,
        "hol_duration_definition": "Span between consecutive positive samples of the same head; gaps are unobserved.",
        "samples_with_unknown_dependencies": unknown_samples,
        "incomplete_spans": len(starts),
        "errors": errors,
        "resource_peaks": resource_peaks,
        "hol_evidence": hol_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.trace.read_text()))
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
