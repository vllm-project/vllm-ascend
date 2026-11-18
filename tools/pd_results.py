# SPDX-License-Identifier: Apache-2.0
"""Combine disjoint, timed AISBench decode windows after real-KV prewarming."""

import math

import pandas as pd


def _milliseconds(value):
    number, unit = str(value).split()
    duration = float(number)
    if unit != "ms" or not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Expected a finite latency in ms, got {value!r}")
    return duration


def aggregate_decode_windows(windows):
    """Return the AISBench CSV/JSON result pair, excluding time spent warming KV.

    Throughput uses exact token counts divided by summed formal durations.
    Means use the number of latency samples in each window. Per-window
    percentiles remain in the native AISBench artifacts.
    """
    total_duration_ms = 0.0
    counts = dict.fromkeys(
        ("Total Requests", "Success Requests", "Failed Requests", "Total Input Tokens", "Total Generated Tokens"), 0
    )
    max_concurrency = 0
    latencies = {"TTFT": [], "TPOT": []}
    for frame, metrics in windows:
        duration_ms = _milliseconds(metrics["Benchmark Duration"]["total"])
        if duration_ms <= 0:
            raise ValueError("Decode window has zero benchmark duration")
        total_duration_ms += duration_ms
        for metric in counts:
            counts[metric] += int(metrics[metric]["total"])
        max_concurrency = max(max_concurrency, int(metrics["Max Concurrency"]["total"]))
        total = frame[frame["Stage"] == "total"]
        for metric, samples in latencies.items():
            row = total.loc[metric]
            n = int(row["N"])
            if n <= 0 or n != int(metrics["Success Requests"]["total"]):
                raise ValueError(f"Invalid {metric} sample count in decode window: {n}")
            samples.append((n, *(_milliseconds(row[key]) for key in ("Average", "Min", "Max"))))
    if not total_duration_ms:
        raise ValueError("No measured decode windows")

    seconds = total_duration_ms / 1000
    generated = counts["Total Generated Tokens"]
    inputs = counts["Total Input Tokens"]
    summary = {key: {"total": value} for key, value in counts.items()}
    summary.update(
        {
            "Benchmark Duration": {"total": f"{total_duration_ms:.4f} ms"},
            "Max Concurrency": {"total": max_concurrency},
            "Request Throughput": {"total": f"{counts['Success Requests'] / seconds:.4f} req/s"},
            "Input Token Throughput": {"total": f"{inputs / seconds:.4f} token/s"},
            "Output Token Throughput": {"total": f"{generated / seconds:.4f} token/s"},
            "Total Token Throughput": {"total": f"{(inputs + generated) / seconds:.4f} token/s"},
        }
    )
    rows = {}
    for metric, samples in latencies.items():
        n = sum(count for count, *_ in samples)
        rows[metric] = {
            "Stage": "total",
            "Average": f"{sum(count * mean for count, mean, _, _ in samples) / n:.4f} ms",
            "Min": f"{min(low for _, _, low, _ in samples):.4f} ms",
            "Max": f"{max(high for _, _, _, high in samples):.4f} ms",
            "N": n,
        }
    frame = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Performance Parameters")
    return [frame, summary]
