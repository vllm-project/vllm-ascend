# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the Ascend A3 KDA length sweep from the published summary CSV."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

LENGTHS = range(1, 17)
PLOT_BATCHES = range(1, 51)
SWEEP_BATCHES = range(1, 81)
LATENCY_TICKS = (1, 10, 20, 30, 40, 50)
MEASURES = (
    "fused_prep_native_us",
    "fused_prep_us",
    "cv_pair_single_us",
    "cv_pair_double_us",
)
COLORS = ("#265ea3", "#087660", "#a85215", "#a5427f")
MISSING_COLOR = "#e5e7eb"


def read_medians(path):
    medians = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"tokens", "batch", "status", *MEASURES}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                tokens, batch = int(row["tokens"]), int(row["batch"])
                if tokens not in LENGTHS or batch not in SWEEP_BATCHES:
                    raise ValueError("expected tokens 1–16 and batch 1–80")
                key = (tokens, batch)
                if key in medians:
                    raise ValueError(f"duplicate shape T={tokens}, B={batch}")
                values = {}
                for field in MEASURES:
                    value = row[field]
                    if value and value.strip():
                        value_us = float(value)
                        if not math.isfinite(value_us) or value_us <= 0:
                            raise ValueError(f"{field} must contain a positive finite median or be blank")
                        values[field] = value_us
                medians[key] = values
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    return medians


def plot_latency(medians, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.115, top=0.83, hspace=0.42, wspace=0.2)
    fig.suptitle("Ascend A3 KDA verification latency", x=0.075, y=0.978, ha="left", weight="bold", fontsize=20)
    fig.text(0.075, 0.925, "Median per full-layer Graph forward across batches 1–50", fontsize=12, color="#475569")
    styles = (
        Line2D([], [], linestyle="--", color="#283444", marker="o", markersize=3, label="Native (paired)"),
        Line2D([], [], linestyle="-", color="#283444", marker="o", markersize=3, label="TileLang fused-prep"),
    )
    fig.legend(handles=styles, loc="upper right", bbox_to_anchor=(0.986, 0.96), frameon=False, ncol=2)

    for index, ax in enumerate(axes.flat):
        first = index * 4 + 1
        ax.set_title(f"{first}–{first + 3} tokens per request", loc="left", weight="bold", pad=9)
        has_values = False
        handles = []
        for tokens, color in zip(range(first, first + 4), COLORS):
            native = []
            fused = []
            for batch in PLOT_BATCHES:
                row = medians.get((tokens, batch), {})
                if "fused_prep_native_us" in row and "fused_prep_us" in row:
                    native.append(row["fused_prep_native_us"] / 1000)
                    fused.append(row["fused_prep_us"] / 1000)
                    has_values = True
                else:
                    native.append(np.nan)
                    fused.append(np.nan)
            markers = {batch - 1 for batch in LATENCY_TICKS}
            for point, value in enumerate(native):
                if math.isfinite(value) and (
                    point == 0
                    or point == len(native) - 1
                    or math.isnan(native[point - 1])
                    or math.isnan(native[point + 1])
                ):
                    markers.add(point)
            marker_style = {"marker": "o", "markersize": 2.2, "markevery": sorted(markers)}
            ax.plot(PLOT_BATCHES, native, color=color, linestyle="--", linewidth=1.35, **marker_style)
            (handle,) = ax.plot(PLOT_BATCHES, fused, color=color, linewidth=1.85, label=f"T{tokens}", **marker_style)
            handles.append(handle)
        if not has_values:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "No matched timings", transform=ax.transAxes, ha="center", color="#64748b")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0.5, 50.5)
        ax.set_xticks(LATENCY_TICKS)
        ax.tick_params(labelbottom=True)
        ax.set(xlabel="Batch size", ylabel="Latency (ms)")
        ax.grid(axis="y", color="#dce4ec", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=9, handlelength=1.1, frameon=False)

    fig.text(
        0.075,
        0.026,
        "T denotes tokens per request. Curves use one point per measured integer batch, with gaps for missing pairs. "
        "Y-axis ranges vary by panel.",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)


def plot_cv_effect(medians, output):
    improvement = np.full((len(LENGTHS), len(SWEEP_BATCHES)), np.nan)
    for (tokens, batch), row in medians.items():
        if "cv_pair_single_us" in row and "cv_pair_double_us" in row:
            improvement[tokens - 1, batch - 1] = 100 * (1 - row["cv_pair_double_us"] / row["cv_pair_single_us"])

    missing = np.isnan(improvement)
    measured = improvement[~missing]
    limit = max(1, math.ceil(np.max(np.abs(measured)))) if measured.size else 1
    cmap = mpl.colormaps["RdBu"].with_extremes(bad=MISSING_COLOR)

    fig, ax = plt.subplots(figsize=(12.8, 6))
    fig.subplots_adjust(left=0.077, right=0.95, bottom=0.23, top=0.81)
    fig.suptitle("Ascend A3 Cube/Vector double-slot effect", x=0.077, y=0.965, ha="left", weight="bold", fontsize=20)
    fig.text(
        0.077,
        0.884,
        "Paired full-layer Graph medians · positive values indicate lower latency with two slots",
        color="#475569",
        fontsize=12,
    )
    cells = ax.imshow(
        improvement,
        origin="lower",
        extent=(0.5, 80.5, 0.5, 16.5),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    if measured.size:
        rows, columns = np.where(missing)
        ax.scatter(columns + 1, rows + 1, marker="x", s=9, color="#64748b", linewidths=0.65)
    ax.axhline(8.5, color="#334155", linestyle="--", linewidth=1.1)
    ax.axvline(50.5, color="#334155", linestyle=":", linewidth=1.35)
    ax.set(
        xlabel="Batch size",
        ylabel="Tokens per request",
        xticks=(*LATENCY_TICKS, 60, 70, 80),
        yticks=list(LENGTHS),
    )
    colorbar = fig.colorbar(cells, ax=ax, fraction=0.034, pad=0.026, format=PercentFormatter(xmax=100))
    colorbar.set_label("Latency reduction")
    fig.legend(
        handles=(
            Patch(facecolor=MISSING_COLOR, edgecolor="#cbd5e1", label="Missing matched pair"),
            Line2D([], [], color="#334155", linestyle="--", label="Projection row boundary at T8/T9"),
            Line2D([], [], color="#334155", linestyle=":", label="Latency chart ends at B50"),
        ),
        loc="lower left",
        bbox_to_anchor=(0.068, 0.073),
        frameon=False,
        ncol=3,
    )
    fig.text(
        0.077,
        0.042,
        f"Reduction = 100 × (1 − double-slot median / single-slot median). Matched shapes: {measured.size:,} / 1,280.",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="length sweep summary with matched median timings")
    parser.add_argument("--out-dir", type=Path, help="output directory (default is beside the CSV)")
    args = parser.parse_args()
    medians = read_medians(args.csv)
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    mpl.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "a3-kda-length-sweep"})
    plot_latency(medians, out_dir / "latency-b1-50.svg")
    plot_cv_effect(medians, out_dir / "cv-slot-effect-b1-80.svg")


if __name__ == "__main__":
    main()
