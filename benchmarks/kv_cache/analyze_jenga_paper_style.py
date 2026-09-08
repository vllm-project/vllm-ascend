#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize two vLLM serving benchmark modes and enforce the quality gate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

FIELDS = (
    "completed",
    "failed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
    "max_concurrent_requests",
)


def _load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Saved results contain metadata plus benchmark fields at the top level in
    # current vLLM.  Accept a nested result as well for revision portability.
    return data.get("result", data)


def _gate_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["choices"][0]["text"]


def _running_timeline(result: dict, step_seconds: float = 0.5) -> list[dict]:
    starts = result.get("start_times", [])
    ttfts = result.get("ttfts", [])
    itls = result.get("itls", [])
    intervals = []
    for start, ttft, request_itls in zip(starts, ttfts, itls):
        end = start + ttft + sum(request_itls)
        intervals.append((start, end))
    if not intervals:
        return []
    origin = min(start for start, _ in intervals)
    end = max(finish for _, finish in intervals)
    timeline = []
    t = origin
    while t <= end + step_seconds:
        timeline.append(
            {
                "seconds": round(t - origin, 3),
                "active_requests": sum(start <= t < finish for start, finish in intervals),
            }
        )
        t += step_seconds
    return timeline


def _parse_metrics(path: Path) -> dict:
    samples: list[dict] = []
    current: dict | None = None
    metric_names = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_usage",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("timestamp_seconds "):
            current = {"timestamp_seconds": float(line.split()[1])}
            samples.append(current)
            continue
        if current is None or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(maxsplit=1)[0]
        for prometheus_name, field in metric_names.items():
            if metric_name != prometheus_name:
                continue
            match = re.search(r"\s([-+0-9.eE]+)$", line)
            if match:
                current[field] = current.get(field, 0.0) + float(match.group(1))
            break
    active_samples = [sample for sample in samples if sample.get("running", 0) > 0]
    kv_samples = [sample["kv_cache_usage"] for sample in active_samples if "kv_cache_usage" in sample]
    running_samples = [sample.get("running", 0.0) for sample in active_samples]
    waiting_samples = [sample.get("waiting", 0.0) for sample in samples]
    return {
        "samples": samples,
        "mean_kv_cache_usage": statistics.fmean(kv_samples) if kv_samples else None,
        "peak_kv_cache_usage": max(kv_samples, default=None),
        "mean_server_running_requests": (statistics.fmean(running_samples) if running_samples else None),
        "peak_server_running_requests": max(running_samples, default=None),
        "peak_server_waiting_requests": max(waiting_samples, default=None),
    }


def _summary(result: dict, metrics_path: Path) -> dict:
    summary = {field: result.get(field) for field in FIELDS}
    timeline = _running_timeline(result)
    active = [point["active_requests"] for point in timeline]
    summary["mean_active_requests"] = statistics.fmean(active) if active else 0.0
    summary["peak_active_requests"] = max(active, default=0)
    summary["timeline"] = timeline
    summary["server_metrics"] = _parse_metrics(metrics_path)
    return summary


def _validate_benchmark_pair(
    workload: str,
    mode_results: dict[str, dict],
) -> None:
    failures: list[str] = []
    for mode, result in mode_results.items():
        expected = result.get("num_prompts")
        completed = result.get("completed")
        failed = result.get("failed")
        if failed != 0 or (expected is not None and completed != expected):
            failures.append(f"{mode}: completed={completed}, failed={failed}, expected={expected}")
    if failures:
        raise SystemExit(f"invalid benchmark {workload}: not every request completed (" + "; ".join(failures) + ")")

    baseline = mode_results["static_partition"]
    addressed = mode_results["address_table"]
    for field in ("input_lens", "output_lens", "generated_texts"):
        if baseline.get(field) != addressed.get(field):
            raise SystemExit(f"correctness gate failed for {workload}: {field} differs between allocation modes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    modes = ("static_partition", "address_table")
    gate_texts = {mode: _gate_text(args.result_root / mode / "correctness-gate.json") for mode in modes}
    gate_match = gate_texts[modes[0]] == gate_texts[modes[1]]
    if not gate_match:
        raise SystemExit("correctness gate failed: allocation modes produced different text")

    common_rate_workloads = set(path.stem for path in (args.result_root / modes[0]).glob("rate-*.json")) & set(
        path.stem for path in (args.result_root / modes[1]).glob("rate-*.json")
    )
    rate_workloads = sorted(
        common_rate_workloads,
        key=lambda workload: float(workload.removeprefix("rate-")),
    )
    workloads = ("throughput-8k", "mixed-memory", *rate_workloads)
    results: dict[str, dict[str, dict]] = {}
    for workload in workloads:
        raw_mode_results: dict[str, dict] = {}
        for mode in modes:
            raw_mode_results[mode] = _load_result(args.result_root / mode / f"{workload}.json")
        _validate_benchmark_pair(workload, raw_mode_results)
        results[workload] = {}
        for mode in modes:
            results[workload][mode] = _summary(
                raw_mode_results[mode],
                args.result_root / mode / f"{workload}-metrics.txt",
            )

    comparisons = {}
    for workload, mode_results in results.items():
        baseline = mode_results["static_partition"]
        addressed = mode_results["address_table"]
        comparisons[workload] = {}
        for metric in (
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_active_requests",
            "peak_active_requests",
        ):
            base = baseline.get(metric)
            new = addressed.get(metric)
            comparisons[workload][f"{metric}_ratio"] = (
                new / base if isinstance(base, (int, float)) and base and isinstance(new, (int, float)) else None
            )
        for metric in (
            "mean_kv_cache_usage",
            "peak_kv_cache_usage",
            "mean_server_running_requests",
            "peak_server_running_requests",
            "peak_server_waiting_requests",
        ):
            base = baseline["server_metrics"].get(metric)
            new = addressed["server_metrics"].get(metric)
            comparisons[workload][f"{metric}_ratio"] = (
                new / base if isinstance(base, (int, float)) and base and isinstance(new, (int, float)) else None
            )
        for metric in ("mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2el_ms", "p99_e2el_ms"):
            base = baseline.get(metric)
            new = addressed.get(metric)
            comparisons[workload][f"{metric}_reduction"] = (
                1 - new / base if isinstance(base, (int, float)) and base and isinstance(new, (int, float)) else None
            )

    output = {
        "correctness_gate": {
            "byte_identical": gate_match,
            "benchmark_outputs_byte_identical": True,
            "text": gate_texts["address_table"],
        },
        "results": results,
        "comparisons": comparisons,
        "methodology_note": (
            "Scaled single-A3 reproduction. Static partition and address-table modes "
            "share the same logical cache interfaces and differ only in physical allocation policy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
