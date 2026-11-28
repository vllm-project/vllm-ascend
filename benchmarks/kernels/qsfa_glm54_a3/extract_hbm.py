#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Extract per-window HBM bandwidth from a MindStudio msprof SQLite export.

The HBM table reports bandwidth in bytes/s and uses STRING_IDS for direction.
https://mindstudio-docs-2610.readthedocs.io/zh-cn/latest/msprof/docs/en/user_guide/profile_data_file_references_db/#hbm
"""

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import median

import regex as re

HBM_COLUMNS = ("deviceId", "timestampNs", "bandwidth", "hbmId", "type")
CLOCKS = ("realtime", "monotonic")
READ_LABELS = {"read", "hbmread", "readbandwidth", "hbmreadbandwidth", "readbw", "hbmreadbw"}
WRITE_LABELS = {"write", "hbmwrite", "writebandwidth", "hbmwritebandwidth", "writebw", "hbmwritebw"}


def table_schema(db, name):
    return {row[1]: row[2] for row in db.execute(f"PRAGMA table_info({name})")}


def hbm_description(db):
    schema = table_schema(db, "HBM")
    missing = set(map(str.lower, HBM_COLUMNS)) - set(map(str.lower, schema))
    if missing:
        raise ValueError(f"HBM table is missing columns {sorted(missing)}. Check --sys-hardware-mem=on and DB export.")

    string_schema = table_schema(db, "STRING_IDS")
    string_columns = {name.lower() for name in string_schema}
    raw_types = [row[0] for row in db.execute("SELECT DISTINCT type FROM HBM")]
    labels = {kind: [] for kind in raw_types}
    if {"id", "value"} <= string_columns:
        for kind in raw_types:
            labels[kind] = [row[0] for row in db.execute("SELECT DISTINCT value FROM STRING_IDS WHERE id=?", (kind,))]
    devices = []
    for device, first, last, count in db.execute(
        "SELECT deviceId, MIN(timestampNs), MAX(timestampNs), COUNT(*) FROM HBM GROUP BY deviceId"
    ):
        devices.append({"device_id": device, "first_ns": first, "last_ns": last, "rows": count})
    units = []
    for device, unit, kind, count in db.execute(
        "SELECT deviceId, hbmId, type, COUNT(*) FROM HBM GROUP BY deviceId, hbmId, type"
    ):
        units.append({"device_id": device, "hbm_id": unit, "type": kind, "rows": count})
    return {
        "hbm_schema": schema,
        "string_ids_schema": string_schema,
        "type_labels": labels,
        "devices": devices,
        "units": units,
        "bandwidth_unit": "bytes/s",
    }


def resolve_directions(kinds, labels, args):
    if len(kinds) != 2 or None in kinds:
        raise ValueError(f"Expected distinct read and write types, found {kinds!r}. Inspect --describe.")
    if args.read_type is not None or args.write_type is not None:
        mapping = {args.read_type: "read", args.write_type: "write"}
        if set(mapping) != set(map(str, kinds)):
            raise ValueError("Provide both --read-type and --write-type using the two raw HBM type values.")
        return {kind: mapping[str(kind)] for kind in kinds}

    result = {}
    for kind in kinds:
        names = {re.sub(r"[^a-z]", "", str(label).lower()) for label in [kind, *labels[kind]]}
        matches = {direction for direction, known in (("read", READ_LABELS), ("write", WRITE_LABELS)) if names & known}
        if len(matches) != 1:
            raise ValueError(
                f"Cannot resolve HBM type {kind!r} from {labels[kind]!r}. "
                "Inspect --describe and pass both type options."
            )
        (result[kind],) = matches
    if set(result.values()) != {"read", "write"}:
        raise ValueError(f"Both directions must be present, resolved {result!r}.")
    return result


def choose_device(description, requested):
    devices = [row["device_id"] for row in description["devices"]]
    if requested is not None and requested in devices:
        return requested
    if requested is None and len(devices) == 1 and devices[0] is not None:
        return devices[0]
    raise ValueError(
        f"Choose an HBM device from {devices!r} using --device-id. "
        "Empty data requires a capture with HBM sampling enabled."
    )


def load_hbm(db, description, device, args):
    kinds = {row["type"] for row in description["units"] if row["device_id"] == device}
    directions = resolve_directions(kinds, description["type_labels"], args)
    series = defaultdict(list)
    for unit, kind, time_ns, bw in db.execute(
        "SELECT hbmId, type, timestampNs, bandwidth FROM HBM WHERE deviceId=? ORDER BY hbmId, type, timestampNs",
        (device,),
    ):
        if type(unit) is not int or type(time_ns) is not int or time_ns < 0:
            raise ValueError(f"HBM access unit and timestamp must be valid integers, got {unit!r} and {time_ns!r}.")
        if not isinstance(bw, (int, float)) or not math.isfinite(bw) or bw < 0:
            raise ValueError(f"Invalid HBM bandwidth {bw!r} at timestamp {time_ns}.")
        samples = series[unit, directions[kind]]
        if samples and samples[-1][0] >= time_ns:
            raise ValueError(f"Duplicate or unordered timestamps for HBM unit {unit}, {directions[kind]}.")
        samples.append((time_ns, bw))

    if not series or any(
        (unit, opposite) not in series
        for unit, opposite in ((unit, "read" if direction == "write" else "write") for unit, direction in series)
    ):
        raise ValueError("Each HBM access unit must supply both read and write samples.")
    return series


def load_windows(path):
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    windows = doc.get("windows") if isinstance(doc, dict) else doc
    if not isinstance(windows, list) or not windows or not all(isinstance(window, dict) for window in windows):
        raise ValueError("Windows JSON must be a nonempty array of objects, or an object with a windows array.")
    return windows, doc.get("clock") if isinstance(doc, dict) else None


def get_window_times(window, clock, declared_clock):
    starts = [f"start_{clock}_ns", f"{clock}_start_ns"]
    ends = [f"end_{clock}_ns", f"{clock}_end_ns"]
    if clock == declared_clock:
        starts.append("start_ns")
        ends.append("end_ns")
    start = next((window[name] for name in starts if name in window), None)
    end = next((window[name] for name in ends if name in window), None)
    if type(start) is int and type(end) is int and 0 <= start < end:
        return start, end
    return None


def align_windows(windows, declared_clock, series, args):
    if args.clock_offset_ns and args.clock == "auto":
        raise ValueError("Specify --clock explicitly when applying a known --clock-offset-ns.")
    candidates = CLOCKS if args.clock == "auto" else (args.clock,)
    common_start = max(samples[0][0] for samples in series.values())
    common_end = min(samples[-1][0] for samples in series.values())
    trim = round(args.trim_ms * 1_000_000)
    matched = {}
    for clock in candidates:
        times = [
            get_window_times(window, clock, declared_clock or (clock if args.clock != "auto" else None))
            for window in windows
        ]
        if any(item is None for item in times):
            continue
        shifted = [(start + args.clock_offset_ns, end + args.clock_offset_ns) for start, end in times]
        if all(common_start <= start + trim < end - trim <= common_end for start, end in shifted):
            matched[clock] = shifted
    if len(matched) != 1:
        raise ValueError(
            f"Cannot select one clock for all trimmed windows, matches {list(matched)}. Common HBM range is "
            f"[{common_start}, {common_end}] ns. Record both absolute nanosecond clocks, "
            "include capture head and tail, and use --clock or a verified --clock-offset-ns if needed."
        )
    clock, times = next(iter(matched.items()))
    chronological = sorted(times)
    if any(before[1] > after[0] for before, after in zip(chronological, chronological[1:])):
        raise ValueError("Measurement windows overlap. Record nonoverlapping synchronized windows.")
    return clock, times, trim


def summarize_window(window, index, bounds, series, trim, args):
    label = window.get("label", window.get("backend"))
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Window {index} requires a nonempty label or backend string.")
    start, end = bounds
    inside = {
        key: [(ts, bw) for ts, bw in records if start + trim <= ts <= end - trim] for key, records in series.items()
    }
    sample_counts = [len(records) for records in inside.values()]
    if min(sample_counts) < args.min_samples:
        raise ValueError(
            f"Window {index} {label!r} has {min(sample_counts)} interior samples per least-sampled series. "
            "Increase capture window length or sampling rate."
        )
    aligned_start = max(records[0][0] for records in inside.values())
    aligned_end = min(records[-1][0] for records in inside.values())
    if aligned_end <= aligned_start:
        raise ValueError(f"Window {index} {label!r} has no common interior interval across all HBM series.")

    byte_totals = {"read": 0.0, "write": 0.0}
    for (unit, direction), records in inside.items():
        intervals = [(t0, t1, bw1) for (t0, _), (t1, bw1) in zip(records, records[1:])]
        cadence = median(t1 - t0 for t0, t1, _ in intervals)
        for t0, t1, bandwidth in intervals:
            duration = min(t1, aligned_end) - max(t0, aligned_start)
            if duration > 0:
                if t1 - t0 > args.max_gap_periods * cadence:
                    raise ValueError(
                        f"Window {index} {label!r} has a sampling gap of {t1 - t0} ns "
                        f"on HBM unit {unit} {direction}, versus median cadence {cadence} ns."
                    )
                byte_totals[direction] += bandwidth * duration / 1e9

    seconds = (aligned_end - aligned_start) / 1e9
    read_rate, write_rate = (byte_totals[direction] / seconds for direction in ("read", "write"))
    output = {
        "index": index,
        "label": label,
        "batch": window.get("batch"),
        "tokens": window.get("tokens"),
        "window_duration_ns": end - start,
        "common_interior_duration_ns": aligned_end - aligned_start,
        "min_series_samples": min(sample_counts),
        "max_series_samples": max(sample_counts),
        "read_bytes_per_second": read_rate,
        "write_bytes_per_second": write_rate,
        "total_bytes_per_second": read_rate + write_rate,
    }
    if args.peak_gb_s is not None:
        output["physical_mbu_pct"] = 100 * output["total_bytes_per_second"] / (1e9 * args.peak_gb_s)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true", help="List DB schema, raw type labels, and time bounds.")
    mode.add_argument(
        "--windows", type=Path, help="JSON of synchronized label and start/end times for each backend window."
    )
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--read-type", help="Known raw read type value, if STRING_IDS cannot resolve it.")
    parser.add_argument("--write-type", help="Known raw write type value, if STRING_IDS cannot resolve it.")
    parser.add_argument("--clock", choices=("auto", *CLOCKS), default="auto")
    parser.add_argument("--clock-offset-ns", type=int, default=0, help="Verified offset to add to window timestamps.")
    parser.add_argument("--trim-ms", type=float, default=50, help="Exclude this many milliseconds at each window edge.")
    parser.add_argument(
        "--min-samples", type=int, default=5, help="Minimum interior sample count for each HBM unit and direction."
    )
    parser.add_argument(
        "--max-gap-periods", type=float, default=3, help="Largest allowed sample gap in median sampling periods."
    )
    parser.add_argument(
        "--peak-gb-s", type=float, help="Optional documented peak HBM bandwidth for one device in decimal GB/s."
    )
    args = parser.parse_args()
    if not math.isfinite(args.trim_ms) or args.trim_ms < 0 or args.min_samples < 2:
        parser.error("--trim-ms must be nonnegative and --min-samples must be at least 2")
    if not math.isfinite(args.max_gap_periods) or args.max_gap_periods < 1:
        parser.error("--max-gap-periods must be at least 1")
    if args.peak_gb_s is not None and (not math.isfinite(args.peak_gb_s) or args.peak_gb_s <= 0):
        parser.error("--peak-gb-s must be positive")
    try:
        with closing(sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            description = hbm_description(db)
            if args.describe:
                print(json.dumps(description, indent=2, allow_nan=False))
                return
            device = choose_device(description, args.device_id)
            series = load_hbm(db, description, device, args)
        windows, declared_clock = load_windows(args.windows)
        clock, bounds, trim = align_windows(windows, declared_clock, series, args)
        report = {
            "clock": clock,
            "clock_offset_ns": args.clock_offset_ns,
            "trim_ns_per_edge": trim,
            "device_id": device,
            "hbm_access_units": sorted({unit for unit, _ in series}),
            "peak_gb_s": args.peak_gb_s,
            "aggregation": (
                "Time-weighted sum of read and write rates across access units on their common interior interval. "
                "Each sample supplies the rate for the preceding interval."
            ),
            "windows": [
                summarize_window(window, index, times, series, trim, args)
                for index, (window, times) in enumerate(zip(windows, bounds), start=1)
            ],
        }
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
