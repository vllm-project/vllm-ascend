#!/usr/bin/env python3
"""Parse msprof op_summary CSVs and compute operator latency.

Groups PROF_* directories by collection timestamp into separate cases.
  --case N   : analyze case N only (1-indexed, sorted by time)
  --latest   : analyze the most recent case (default)
  --all      : analyze all cases
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime
from statistics import median

_DEFAULT_PROF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prof"
)
KERNEL_SUBSTR = "AllGatherQuantMatmulKernel"
GROUP_THRESHOLD_S = 5

_TS_RE = re.compile(r"PROF_\d+_(\d{17})_")


def parse_ts_from_dir(dirname):
    m = _TS_RE.search(dirname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S%f")


def find_csvs(prof_base, prof_dir_name):
    return glob.glob(
        os.path.join(prof_base, prof_dir_name,
                     "mindstudio_profiler_output", "op_summary_*.csv")
    )


def parse_csv(csv_path):
    device_durations = {}
    device_cubes = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if KERNEL_SUBSTR not in row.get("Op Name", ""):
                continue
            dev = row["Device_id"].strip()
            try:
                dur = float(row["Task Duration(us)"].strip())
            except (KeyError, ValueError):
                continue
            device_durations.setdefault(dev, []).append(dur)
            try:
                cube = float(row["cube_utilization(%)"].strip())
            except (KeyError, ValueError):
                cube = 0.0
            device_cubes.setdefault(dev, []).append(cube)
    return device_durations, device_cubes


def group_cases(prof_dirs):
    prof_dirs_sorted = sorted(prof_dirs, key=parse_ts_from_dir)
    cases = []
    current_group = []
    for d in prof_dirs_sorted:
        ts = parse_ts_from_dir(d)
        if not current_group or (ts - parse_ts_from_dir(current_group[0])).total_seconds() < GROUP_THRESHOLD_S:
            current_group.append(d)
        else:
            cases.append(current_group)
            current_group = [d]
    if current_group:
        cases.append(current_group)
    return cases


def read_case_info(prof_base, case_dirs):
    for d in case_dirs:
        info_path = os.path.join(prof_base, d, "case_info.txt")
        if os.path.isfile(info_path):
            with open(info_path) as f:
                return f.read().strip()
    return None


def read_case_precision(prof_base, case_dirs):
    for d in case_dirs:
        prec_path = os.path.join(prof_base, d, "case_precision.txt")
        if os.path.isfile(prec_path):
            with open(prec_path) as f:
                return f.read().strip()
    return None


def read_case_info_from_input(prof_base, case_dirs, mkn_str=None):
    info = read_case_info(prof_base, case_dirs)
    if info:
        return info
    if mkn_str:
        parts = mkn_str.split()
        if len(parts) >= 3:
            return f"M={parts[0]} K={parts[1]} N={parts[2]}"
        return f"M={mkn_str}"
    return None


def report_case(prof_base, case_dirs, case_idx, mkn_str=None):
    all_devices = {}
    all_cubes = {}
    for prof_dir in case_dirs:
        for csv_path in find_csvs(prof_base, prof_dir):
            durs, cubes = parse_csv(csv_path)
            for dev, dv in durs.items():
                all_devices.setdefault(dev, []).extend(dv)
            for dev, cv in cubes.items():
                all_cubes.setdefault(dev, []).extend(cv)

    if not all_devices:
        print(f"  [Case {case_idx}] No AllGatherQuantMatmulKernel rows found.\n")
        return None

    info = read_case_info_from_input(prof_base, case_dirs, mkn_str)
    info_line = f"  [{info}]" if info else ""

    medians = []
    cube_medians = []
    print(f"  [Case {case_idx}]  ({len(case_dirs)} ranks, {case_dirs[0]})")
    if info_line:
        print(f"  {info_line}")
    print("  " + "-" * 58)
    for dev in sorted(all_devices.keys(), key=lambda x: int(x)):
        durs = sorted(all_devices[dev])
        med = median(durs)
        medians.append(med)
        cube_vals = all_cubes.get(dev, [0.0])
        cube_med = median(cube_vals)
        cube_medians.append(cube_med)
        print(
            f"    Rank {dev:4s} | {len(durs):3d} iters | "
            f"median={med:.3f} us | min={durs[0]:.3f} us | max={durs[-1]:.3f} us | "
            f"cube={cube_med:.1f}%"
        )
        for i, d in enumerate(durs):
            print(f"                 iter[{i:2d}] = {d:.3f} us")

    avg = sum(medians) / len(medians)
    cube_avg = sum(cube_medians) / len(cube_medians) if cube_medians else 0.0
    print("  " + "-" * 58)
    print(
        f"    Overall latency (avg of card medians): "
        f"{avg:.3f} us  ({len(medians)} cards)"
    )
    print(f"    Cube utilization (avg of card medians): {cube_avg:.1f}%")

    precision = read_case_precision(prof_base, case_dirs)
    if precision:
        print(f"    Precision: {precision}")
    print()
    return avg, info, precision, cube_avg


def print_summary(results):
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    header = f"  {'Case':>4s}  {'M':>5s}  {'K':>5s}  {'N':>5s}  {'Latency(us)':>12s}  {'Cube(%)':>7s}  {'Precision':<8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for case_idx, info, avg, precision, cube_avg in results:
        parts = dict()
        if info:
            for kv in info.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    parts[k] = v
        m = parts.get("M", "N/A")
        k = parts.get("K", "N/A")
        n = parts.get("N", "N/A")
        prec = precision if precision else "N/A"
        print(f"  {case_idx:4d}  {m:>5s}  {k:>5s}  {n:>5s}  {avg:12.3f}  {cube_avg:7.1f}  {prec:<8s}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Parse msprof op_summary CSVs.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--case", type=int, help="Case index (1-based, sorted by time)")
    grp.add_argument("--latest", action="store_true", default=True,
                     help="Analyze the most recent case (default)")
    grp.add_argument("--all", action="store_true", help="Analyze all cases")
    grp.add_argument("--check-latest-threshold", type=float, default=None,
                     help="Check latest case's latency against threshold. Print only the latency number. Exit 0 if <= threshold, exit 2 if > threshold.")
    parser.add_argument("--mkn", type=str, default=None,
                        help="M K N string, e.g. '1010 4096 2560'")
    parser.add_argument("--prof-dir", default=_DEFAULT_PROF,
                        help="prof/ directory (default: auto-detect)")
    args = parser.parse_args()
    prof_base = os.path.abspath(args.prof_dir)

    prof_dirs = [
        d for d in os.listdir(prof_base)
        if re.match(r"PROF_\d+_\d{17}_", d)
    ]
    if not prof_dirs:
        print(f"No PROF directories found under {prof_base}", file=sys.stderr)
        return 1

    cases = group_cases(prof_dirs)

    if args.check_latest_threshold is not None:
        idx = len(cases)
        ret = report_case(prof_base, cases[-1], idx, args.mkn)
        if ret:
            avg, info, precision, cube_avg = ret
            print(f"{avg:.3f}")
            if avg <= args.check_latest_threshold:
                return 0
            else:
                return 2
        return 1

    print(f"Found {len(cases)} case(s) in {prof_base}\n")

    print("=" * 64)
    print("AllGatherQuantMatmul — Per-Card Latency (median of iterations)")
    print("=" * 64)

    results = []
    if args.all:
        for i, case_dirs in enumerate(cases, 1):
            ret = report_case(prof_base, case_dirs, i, args.mkn)
            if ret:
                avg, info, precision, cube_avg = ret
                results.append((i, info, avg, precision, cube_avg))
    elif args.case is not None:
        if args.case < 1 or args.case > len(cases):
            print(f"--case {args.case} out of range (1..{len(cases)})",
                  file=sys.stderr)
            return 1
        ret = report_case(prof_base, cases[args.case - 1], args.case, args.mkn)
        if ret:
            avg, info, precision, cube_avg = ret
            results.append((args.case, info, avg, precision, cube_avg))
    else:
        idx = len(cases)
        ret = report_case(prof_base, cases[-1], idx, args.mkn)
        if ret:
            avg, info, precision, cube_avg = ret
            results.append((idx, info, avg, precision, cube_avg))

    if results:
        print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
