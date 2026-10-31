# SPDX-License-Identifier: Apache-2.0
"""Summarize complete warmup-delimited msprof graph sequences without trimming."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import regex as re


def summarize(path, mode, cube_pattern, vector_pattern):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        duration = next((s for s in ("Task Duration(us)", "Task Duration", "Duration(us)") if s in columns), None)
        start = next((s for s in ("Task Start Time(us)", "Task Start Time") if s in columns), None)
        names = [s for s in ("Op Name", "Op Type", "Task Name", "Kernel Name") if s in columns]
        if duration is None or start is None or not names:
            raise ValueError(f"unsupported msprof columns: {columns}")
        rows = sorted(reader, key=lambda row: float(row[start]))
    samples, active = [], None
    for row in rows:
        name = " ".join(row[column] for column in names)
        if re.search("argmax", name, re.IGNORECASE):
            if active is not None:
                samples.append(active)
            active = []
        elif active is not None:
            value = float(row[duration])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid task duration")
            if mode == "fused":
                stage = "fused"
            elif re.search(vector_pattern, name):
                stage = "vector"
            elif re.search(cube_pattern, name):
                stage = "cube"
            else:
                raise ValueError(f"unclassified replay task; inspect raw evidence: {name}")
            active.append({"name": name, "duration_us": value, "stage": stage})
    if active is not None:
        samples.append(active)
    if len(samples) != 10 or any(not s for s in samples):
        raise ValueError(f"expected ten nonempty warmup/target pairs, got {len(samples)}")
    signatures = [[(t["name"], t["stage"]) for t in sample] for sample in samples]
    if any(s != signatures[0] for s in signatures):
        raise ValueError("task sequences differ; retain raw evidence and investigate")
    totals = [sum(t["duration_us"] for t in sample) for sample in samples]
    result = {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_file": str(path),
        "mode": mode,
        "samples": samples,
        "total_us": totals,
        "median_total_us": statistics.median(totals),
        "sample_count": 10,
        "tasks_per_sample": len(samples[0]),
    }
    if mode == "serial":
        cube = [sum(t["duration_us"] for t in sample if t["stage"] == "cube") for sample in samples]
        vector = [total - c for total, c in zip(totals, cube)]
        if min(cube) <= 0 or min(vector) <= 0:
            raise ValueError("every replay must include Cube and Vector tasks")
        result.update(
            cube_us=cube,
            vector_us=vector,
            cube_share=[c / total for c, total in zip(cube, totals)],
            median_cube_share=statistics.median(c / total for c, total in zip(cube, totals)),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--mode", choices=("fused", "serial"), required=True)
    p.add_argument("--cube-pattern", default="(?i)(matmul|gemm)")
    p.add_argument("--vector-pattern", default="serial_(cast|pool|finalize)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite profiling evidence")
    result = summarize(args.csv, args.mode, args.cube_pattern, args.vector_pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "total_us")}))


if __name__ == "__main__":
    main()
