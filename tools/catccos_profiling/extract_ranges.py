#!/usr/bin/env python
"""Aggregate vLLM-Ascend and CatCCOS profiler ranges without erasing M."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

RANGE_PREFIXES = ("vllm_ascend.moe.", "catccos.a5.binding.")


class RangeSample(NamedTuple):
    name: str
    host_us: float
    device_us: float


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, index)]


def read_samples(csv_path: Path) -> list[RangeSample]:
    samples = []
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            name = row.get("Name", "")
            if not name.startswith(RANGE_PREFIXES):
                continue
            try:
                host_us = float(row.get("Host Total Duration(us)", "0") or 0)
                device_us = float(row.get("Device Total Duration(us)", "0") or 0)
            except ValueError:
                continue
            samples.append(RangeSample(name, host_us, device_us))
    return samples


def summarize(samples: list[RangeSample]) -> list[dict[str, str | int | float]]:
    grouped: dict[str, list[RangeSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.name].append(sample)

    rows = []
    for name, group in sorted(grouped.items()):
        host = [sample.host_us for sample in group if sample.host_us > 0]
        device = [sample.device_us for sample in group if sample.device_us > 0]
        rows.append(
            {
                "name": name,
                "count": len(group),
                "host_p50_us": statistics.median(host) if host else 0.0,
                "host_p95_us": percentile(host, 0.95),
                "host_max_us": max(host, default=0.0),
                "device_p50_us": statistics.median(device) if device else 0.0,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Profile roots or operator_details.csv files.",
    )
    parser.add_argument("--output", type=Path, help="Optional aggregate CSV path.")
    return parser.parse_args()


def find_csv_files(roots: list[Path]) -> list[Path]:
    paths = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(root.rglob("operator_details.csv"))
    return sorted(set(paths))


def main() -> None:
    args = parse_args()
    csv_files = find_csv_files(args.roots)
    if not csv_files:
        raise SystemExit("No operator_details.csv files found")

    all_rows = []
    for csv_path in csv_files:
        rows = summarize(read_samples(csv_path))
        print(f"\n# {csv_path}")
        for row in rows:
            print(
                f"{row['count']:5d} "
                f"p50={row['host_p50_us']:9.3f}us "
                f"p95={row['host_p95_us']:9.3f}us "
                f"max={row['host_max_us']:9.3f}us "
                f"{row['name']}"
            )
            all_rows.append({"source": str(csv_path), **row})

    if args.output and all_rows:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote: {args.output}")
    elif args.output:
        raise SystemExit("No matching profiler ranges found")


if __name__ == "__main__":
    main()
