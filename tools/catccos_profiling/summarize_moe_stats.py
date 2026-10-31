#!/usr/bin/env python3
"""Summarize shape-qualified MoE profiler ranges and component shares."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import regex as re

RANGE_PREFIXES = ("vllm_ascend.moe.", "catccos.a5.binding.")
GLOBAL_M_RE = re.compile(r"global_M=(\d+)")
RANK_M_RE = re.compile(r"rank_M=(\d+)")
M_RE = re.compile(r"(?:^|\[)M=(\d+)")


@dataclass(frozen=True)
class Sample:
    backend: str
    stage: str
    global_m: int
    rank_m: int
    host_us: float
    device_us: float
    source: str


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def parse_name(name: str, tp_size: int) -> tuple[str, str, int, int] | None:
    if name.startswith("vllm_ascend.moe."):
        body = name.removeprefix("vllm_ascend.moe.")
        backend, _, metadata = body.partition("[")
        stage = backend.split(".", 1)[1] if "." in backend else ""
        backend = backend.split(".", 1)[0]
    elif name.startswith("catccos.a5.binding."):
        body = name.removeprefix("catccos.a5.binding.")
        stage, _, metadata = body.partition("[")
        backend = "catccos"
    else:
        return None

    global_match = GLOBAL_M_RE.search(metadata)
    rank_match = RANK_M_RE.search(metadata)
    m_match = M_RE.search(metadata)
    if global_match:
        global_m = int(global_match.group(1))
    elif m_match:
        global_m = int(m_match.group(1)) * tp_size
    else:
        return None
    if rank_match:
        rank_m = int(rank_match.group(1))
    elif m_match:
        rank_m = int(m_match.group(1))
    else:
        rank_m = global_m // max(tp_size, 1)

    # Older ranges used only M.  The common host body and the prepare range
    # are entered with the global token tensor.  AllGather keeps that tensor
    # global as well, while MC2/All-to-All and CatCCOS receive a TP-local
    # slice.  Keep this distinction so component shares use the right parent.
    if not global_match and m_match:
        legacy_m = int(m_match.group(1))
        is_global_range = stage in {"host_moe_body", "synchronized_moe_body", "prepare"}
        is_global_range = is_global_range or backend == "native_allgather"
        global_m = legacy_m if is_global_range else legacy_m * tp_size
        rank_m = legacy_m
    return backend, stage, global_m, rank_m


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except (TypeError, ValueError):
        return 0.0


def read_samples(csv_path: Path, tp_size: int) -> list[Sample]:
    samples: list[Sample] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            name = row.get("Name", "")
            if not name.startswith(RANGE_PREFIXES):
                continue
            parsed = parse_name(name, tp_size)
            if parsed is None:
                continue
            backend, stage, global_m, rank_m = parsed
            samples.append(
                Sample(
                    backend=backend,
                    stage=stage,
                    global_m=global_m,
                    rank_m=rank_m,
                    host_us=_float(row, "Host Total Duration(us)"),
                    device_us=_float(row, "Device Total Duration(us)"),
                    source=str(csv_path),
                )
            )
    return samples


def find_csv_files(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(root.rglob("operator_details.csv"))
    return sorted(set(paths))


def summarize(samples: list[Sample]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, int, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.backend, sample.stage, sample.global_m, sample.rank_m)].append(sample)

    rows: list[dict[str, str | int | float]] = []
    for (backend, stage, global_m, rank_m), group in sorted(grouped.items()):
        host = [sample.host_us for sample in group if sample.host_us > 0]
        device = [sample.device_us for sample in group if sample.device_us > 0]
        rows.append(
            {
                "backend": backend,
                "stage": stage,
                "global_M": global_m,
                "rank_M": rank_m,
                "count": len(group),
                "host_p50_us": statistics.median(host) if host else 0.0,
                "host_p95_us": percentile(host, 0.95),
                "host_max_us": max(host, default=0.0),
                "device_p50_us": statistics.median(device) if device else 0.0,
                "device_p95_us": percentile(device, 0.95),
                "device_max_us": max(device, default=0.0),
                "source_count": len({sample.source for sample in group}),
            }
        )

    parent_by_shape: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row["stage"] not in {"host_moe_body", "synchronized_moe_body"}:
            continue
        key = (str(row["backend"]), int(row["global_M"]))
        current = parent_by_shape.get(key)
        if current is None or row["stage"] == "synchronized_moe_body":
            parent_by_shape[key] = {
                "host": float(row["host_p50_us"]),
                "device": float(row["device_p50_us"]),
            }

    for row in rows:
        parent = parent_by_shape.get((str(row["backend"]), int(row["global_M"])))
        host_parent = parent["host"] if parent else 0.0
        device_parent = parent["device"] if parent else 0.0
        row["host_share_of_body_pct"] = 100.0 * float(row["host_p50_us"]) / host_parent if host_parent > 0 else 0.0
        row["device_share_of_body_pct"] = (
            100.0 * float(row["device_p50_us"]) / device_parent if device_parent > 0 else 0.0
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Profile roots or operator_details.csv files.")
    parser.add_argument("--tp-size", type=int, default=4, help="TP size used to infer M for old ranges.")
    parser.add_argument("--output", type=Path, help="Optional summary CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tp_size < 1:
        raise SystemExit("--tp-size must be positive")
    csv_files = find_csv_files(args.roots)
    if not csv_files:
        raise SystemExit("No operator_details.csv files found")
    rows = summarize([sample for path in csv_files for sample in read_samples(path, args.tp_size)])
    if not rows:
        raise SystemExit("No shape-qualified MoE ranges found")

    fieldnames = list(rows[0].keys())
    for row in rows:
        print(
            f"{row['backend']:18s} global_M={row['global_M']:5d} rank_M={row['rank_M']:5d} "
            f"{row['stage']:34s} count={row['count']:4d} "
            f"host_p50={row['host_p50_us']:10.3f}us "
            f"device_p50={row['device_p50_us']:10.3f}us "
            f"host_share={row['host_share_of_body_pct']:7.2f}% "
            f"device_share={row['device_share_of_body_pct']:7.2f}%"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
