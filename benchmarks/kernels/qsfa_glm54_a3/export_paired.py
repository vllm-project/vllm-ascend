# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a complete QSFA paired sweep and export anonymous numeric results."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

BATCHES = range(1, 81)
KEY_BATCHES = (1, 8, 16, 32, 64, 80)
QUERY_TOKENS = 6
IMPLEMENTATIONS = ("native", "candidate")
WORKLOAD = ("valid_sparse_pairs", "useful_flops", "logical_io_bytes")
SPEEDUP = ("paired_speedup", "speedup_ci95_low", "speedup_ci95_high")
GRAPH_POSITIVE = ("native_us", "candidate_us", *SPEEDUP)
GRAPH_NONNEGATIVE = (
    "native_useful_mfu_pct",
    "candidate_useful_mfu_pct",
    "native_logical_mbu_pct",
    "candidate_logical_mbu_pct",
    "native_golden_max_abs",
    "candidate_golden_max_abs",
    "candidate_native_max_abs",
)
GRAPH_FIELDS = ("batch", "tokens", *WORKLOAD, *GRAPH_POSITIVE, *GRAPH_NONNEGATIVE)
HBM_INTEGER = ("hbm_samples", "min_counter_samples")
HBM_FLOAT = (
    "forwards_per_s",
    "hbm_read_gb_s",
    "hbm_write_gb_s",
    "hbm_total_gb_s",
    "physical_mbu_pct",
    "hbm_bytes_per_forward_est",
    "window_range_pp",
)
HBM_FIELDS = (*HBM_INTEGER, *HBM_FLOAT, "hbm_bytes_over_logical")
PROFILE_SHAPE = (
    "context_length",
    "heads",
    "topk",
    "rope_dim",
    "index_pattern",
    "native_impl",
    "scale",
    "seed",
    "peak_bf16_tflops",
    "peak_hbm_gb_s",
)
SUMMARY_GRAPH_FIELDS = {
    "latency_us": "us",
    "useful_mfu_pct": "useful_mfu_pct",
    "logical_mbu_pct": "logical_mbu_pct",
    "golden_max_abs": "golden_max_abs",
}


def read_table(path, required, source):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{source} has duplicated columns")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{source} is missing columns {', '.join(sorted(missing))}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{source} row {reader.line_num} has an incorrect number of cells")
            yield row


def numeric(row, field, kind, minimum, source):
    try:
        value = kind(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{source} has an invalid numeric value for {field}") from None
    if (kind is float and not math.isfinite(value)) or value < minimum:
        raise ValueError(f"{source} has an out-of-range numeric value for {field}")
    return value


def batch_key(row, source):
    batch = numeric(row, "batch", int, 1, source)
    if batch not in BATCHES:
        raise ValueError(f"{source} has a batch outside B1–80")
    return batch


def graph_data(path):
    records, selected = {}, {}
    for row in read_table(path, GRAPH_FIELDS, "graph"):
        batch = batch_key(row, "graph")
        source = f"graph B{batch}"
        if batch in records:
            raise ValueError(f"{source} occurs more than once")
        record = {field: numeric(row, field, int, 1, source) for field in ("batch", "tokens", *WORKLOAD)}
        if record["tokens"] != QUERY_TOKENS * batch:
            raise ValueError(f"{source} has an incorrect number of MTP5 tokens")
        record.update({field: numeric(row, field, float, 0, source) for field in (*GRAPH_POSITIVE, *GRAPH_NONNEGATIVE)})
        if any(record[field] <= 0 for field in GRAPH_POSITIVE):
            raise ValueError(f"{source} needs two positive graph latencies and paired speedup/CI")
        if not record["speedup_ci95_low"] <= record["paired_speedup"] <= record["speedup_ci95_high"]:
            raise ValueError(f"{source} has an invalid paired speedup confidence interval")
        records[batch], selected[batch] = record, row.get("selected")
    missing = set(BATCHES) - records.keys()
    if missing:
        raise ValueError("graph is missing " + ", ".join(f"B{batch}" for batch in sorted(missing)))
    return records, selected


def verified_hbm_candidate(graph_path, hbm_path):
    timing_path = graph_path.with_name("settings.json")
    profile_path = hbm_path.with_name("profile_settings.json")
    if not timing_path.is_file() or not profile_path.is_file():
        return None
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(timing, dict) or not isinstance(profile, dict):
        raise ValueError("graph and profile settings must be JSON objects")
    if profile.get("mode") != "hbm":
        raise ValueError("the profile settings do not describe an HBM capture")
    mismatches = [field for field in PROFILE_SHAPE if field not in timing or timing[field] != profile.get(field)]
    if timing.get("baseline_candidate") != profile.get("baseline_candidate"):
        mismatches.append("baseline_candidate")
    if mismatches:
        raise ValueError(f"graph/HBM workload settings differ on {', '.join(mismatches)}")
    candidate = profile.get("candidate")
    tiles = profile.get("block_i") or []
    if not isinstance(candidate, str) or not candidate or not isinstance(tiles, list) or len(tiles) > 1:
        raise ValueError("the HBM capture must describe one candidate schedule")
    if tiles:
        candidate += f"[block_i={tiles[0]}]"
    if profile.get("packed_gather"):
        candidate += "/packed-gather"
    return candidate


def check_summary_row(row, graph, implementation, source):
    for field in WORKLOAD:
        if numeric(row, field, int, 1, source) != graph[field]:
            raise ValueError(f"{source} disagrees with the graph on {field}")
    expected = {
        field: graph[f"{implementation}_{graph_suffix}"] for field, graph_suffix in SUMMARY_GRAPH_FIELDS.items()
    }
    if implementation == "candidate":
        expected.update({field: graph[field] for field in (*SPEEDUP, "candidate_native_max_abs")})
    for field, value in expected.items():
        if numeric(row, field, float, 0, source) != value:
            raise ValueError(f"{source} disagrees with the graph on {field}")


def hbm_data(path, graph, selected, *, joined, hbm_candidate):
    source = "summary" if joined else "HBM"
    sample_field = "hbm_samples" if joined else "samples"
    required = ("batch", "tokens", "implementation", sample_field, "min_counter_samples", *HBM_FLOAT)
    if joined:
        required += ("operator", *WORKLOAD, *SUMMARY_GRAPH_FIELDS, *SPEEDUP, "candidate_native_max_abs")
    records, seen = {}, set()
    for row in read_table(path, required, source):
        batch = batch_key(row, source)
        implementation = row["implementation"]
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"{source} B{batch} has an unknown implementation")
        label = f"{source} B{batch} {implementation}"
        key = batch, implementation
        if key in seen:
            raise ValueError(f"{label} occurs more than once")
        seen.add(key)
        benchmark = graph[batch]
        if numeric(row, "tokens", int, 1, label) != benchmark["tokens"]:
            raise ValueError(f"{label} disagrees with the graph on tokens")
        if joined:
            check_summary_row(row, benchmark, implementation, label)
            if not any(row[field] for field in (sample_field, "min_counter_samples", *HBM_FLOAT)):
                continue
        if implementation == "candidate":
            candidate = row["operator"] if joined and selected[batch] is not None else hbm_candidate
            if candidate is not None and candidate != selected[batch]:
                raise ValueError(f"{label} has a different candidate schedule")
        record = {
            "hbm_samples": numeric(row, sample_field, int, 1, label),
            "min_counter_samples": numeric(row, "min_counter_samples", int, 1, label),
            **{field: numeric(row, field, float, 0, label) for field in HBM_FLOAT},
        }
        if record["forwards_per_s"] <= 0:
            raise ValueError(f"{label} needs a positive forward rate")
        record["hbm_bytes_over_logical"] = record["hbm_bytes_per_forward_est"] / benchmark["logical_io_bytes"]
        records[key] = record
    return records


def export(graph, hbm, *, hbm_verified, require_full_hbm, destination):
    paired_hbm = sum(all((batch, name) in hbm for name in IMPLEMENTATIONS) for batch in BATCHES)
    if require_full_hbm and paired_hbm != len(BATCHES):
        raise ValueError(f"physical HBM covers both implementations in {paired_hbm}/80 batches")
    rows = [dict(graph[batch]) for batch in BATCHES]
    for (batch, implementation), measured in hbm.items():
        rows[batch - 1].update({f"{implementation}_{key}": value for key, value in measured.items()})

    key_rows = [rows[batch - 1] for batch in KEY_BATCHES]
    ratios = [record["paired_speedup"] for record in rows]
    candidate_wins = sum(ratio > 1 for ratio in ratios)
    native_wins = sum(ratio < 1 for ratio in ratios)
    candidate_ci = sum(row["speedup_ci95_low"] > 1 for row in rows)
    native_ci = sum(row["speedup_ci95_high"] < 1 for row in rows)
    summary = {
        "timed_batches": len(rows),
        "median_batch_speedup": statistics.median(ratios),
        "min_batch_speedup": min(ratios),
        "max_batch_speedup": max(ratios),
        "winner_counts": {
            "candidate": candidate_wins,
            "native": native_wins,
            "tie": len(rows) - candidate_wins - native_wins,
        },
        "ci95_counts": {
            "above_one": candidate_ci,
            "below_one": native_ci,
            "includes_one": len(rows) - candidate_ci - native_ci,
        },
        "hbm_coverage": {
            "native_batches": sum((batch, "native") in hbm for batch in BATCHES),
            "candidate_batches": sum((batch, "candidate") in hbm for batch in BATCHES),
            "paired_batches": paired_hbm,
            "settings_verified": int(hbm_verified),
        },
        "key_batches": key_rows,
    }
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    fields = GRAPH_FIELDS
    if hbm:
        fields += tuple(f"{implementation}_{field}" for implementation in IMPLEMENTATIONS for field in HBM_FIELDS)
    destination.mkdir(parents=True, exist_ok=False)
    for filename, table in (("paired.csv", rows), ("key_batches.csv", key_rows)):
        with (destination / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    (destination / "summary.json").write_text(encoded, encoding="utf-8")
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="benchmark graph.csv")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--hbm", type=Path, help="HBM capture directory or its hbm.csv")
    profile.add_argument("--summary", type=Path, help="joined CSV from summarize.py")
    parser.add_argument("--require-full-hbm", action="store_true", help="require both HBM rows at every batch")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the numeric export")
    args = parser.parse_args()
    try:
        graph, selected = graph_data(args.graph)
        hbm, verified_candidate = {}, None
        if args.hbm:
            path = args.hbm / "hbm.csv" if args.hbm.is_dir() else args.hbm
            verified_candidate = verified_hbm_candidate(args.graph, path)
            hbm = hbm_data(path, graph, selected, joined=False, hbm_candidate=verified_candidate)
        if args.summary:
            hbm = hbm_data(args.summary, graph, selected, joined=True, hbm_candidate=None)
        encoded = export(
            graph,
            hbm,
            hbm_verified=verified_candidate is not None,
            require_full_hbm=args.require_full_hbm,
            destination=args.output,
        )
    except FileExistsError:
        parser.error("the export directory already exists")
    except (OSError, UnicodeError, csv.Error):
        parser.error("an input or output file could not be processed")
    except ValueError as exc:
        parser.error(str(exc))
    print(encoded, end="")


if __name__ == "__main__":
    main()
