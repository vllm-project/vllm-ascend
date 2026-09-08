#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Create reproducible figures for the scaled Jenga-style NPU experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "static_partition": "#64748b",
    "address_table": "#0f766e",
}
LABELS = {
    "static_partition": "Static partition",
    "address_table": "Dynamic address table",
}


def _active_metric_series(samples: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    active_indices = [
        index for index, sample in enumerate(samples) if sample.get("running", 0) > 0 or sample.get("waiting", 0) > 0
    ]
    if not active_indices:
        return np.array([]), {}
    start = max(active_indices[0] - 2, 0)
    end = min(active_indices[-1] + 3, len(samples))
    selected = samples[start:end]
    origin = selected[0]["timestamp_seconds"]
    seconds = np.array([sample["timestamp_seconds"] - origin for sample in selected])
    values = {
        field: np.array([sample.get(field, 0.0) for sample in selected])
        for field in ("running", "waiting", "kv_cache_usage")
    }
    return seconds, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    correctness_gate = data.get("correctness_gate", {})
    if not correctness_gate.get("byte_identical") or not correctness_gate.get("benchmark_outputs_byte_identical"):
        raise SystemExit("refusing to plot results that did not pass both deterministic correctness gates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("static_partition", "address_table")

    # Workload-level request throughput.
    workloads = list(data["results"])
    x = np.arange(len(workloads))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, mode in enumerate(modes):
        values = [data["results"][workload][mode]["request_throughput"] for workload in workloads]
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x, workloads, rotation=20, ha="right")
    ax.set_ylabel("Completed requests / second")
    ax.set_title("Qwen3.5-27B serving throughput on one A3 die")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "throughput_comparison.png", dpi=180)
    plt.close(fig)

    # Figure-15-style request-rate sweep.
    rate_workloads = [name for name in workloads if name.startswith("rate-")]
    rates = np.array([float(name.removeprefix("rate-")) for name in rate_workloads])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    metrics = (
        ("request_throughput", "Throughput (req/s)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_e2el_ms", "Mean E2EL (ms)"),
    )
    for ax, (metric, title) in zip(axes.flat, metrics):
        for mode in modes:
            values = [data["results"][name][mode][metric] for name in rate_workloads]
            ax.plot(
                rates,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xticks(rates)
    axes[1, 0].set_xlabel("Offered request rate (req/s)")
    axes[1, 1].set_xlabel("Offered request rate (req/s)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Poisson request-rate sweep (2K real-text prompts)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rate_sweep_latency.png", dpi=180)
    plt.close(fig)

    # Server-side memory, running batch, and queue timeline for the mixed trace.
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fields = (
        ("running", "Running requests"),
        ("waiting", "Waiting requests"),
        ("kv_cache_usage", "KV cache usage (%)"),
    )
    for mode in modes:
        samples = data["results"]["mixed-memory"][mode]["server_metrics"]["samples"]
        seconds, series = _active_metric_series(samples)
        for ax, (field, ylabel) in zip(axes, fields):
            values = series[field]
            if field == "kv_cache_usage":
                values = values * 100
            ax.step(
                seconds,
                values,
                where="post",
                linewidth=1.8,
                color=COLORS[mode],
                label=LABELS[mode],
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Seconds from first active request")
    fig.suptitle("Mixed 512 / 2K / 8K / 32K workload timeline")
    fig.tight_layout()
    fig.savefig(args.output_dir / "mixed_workload_timeline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
