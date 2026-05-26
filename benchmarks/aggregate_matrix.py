#!/usr/bin/env python3
"""
aggregate_matrix.py — aggregate bench_apc_matrix.sh per-group summary.json
into a single markdown comparison table.

Usage:
    python3 aggregate_matrix.py <results_dir>
    # prints markdown to stdout

Expected layout under <results_dir>:
    G1_align/  config.json  summary.json
    G1_all/    config.json  summary.json
    G2_align/  ...
    ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def load_run(group_dir: Path) -> dict[str, Any] | None:
    cfg_p = group_dir / "config.json"
    sum_p = group_dir / "summary.json"
    failed_p = group_dir / "FAILED"
    if not cfg_p.exists():
        return None
    try:
        cfg = json.loads(cfg_p.read_text())
    except Exception:
        return None
    rec: dict[str, Any] = {
        "group_id": cfg.get("group_id"),
        "mode": cfg.get("mode"),
        "prefix_len": cfg.get("prefix_len"),
        "K": cfg.get("K"),
        "num_prefixes": cfg.get("num_prefixes"),
        "total_prompts": cfg.get("total_prompts"),
        "concurrency": cfg.get("concurrency"),
        "status": "failed" if failed_p.exists() else "ok",
    }
    if sum_p.exists():
        try:
            s = json.loads(sum_p.read_text())
        except Exception:
            s = {}
    else:
        s = {}
        rec["status"] = "missing"

    # vllm bench serve summary.json field names (best effort across versions)
    def pick(*names: str) -> Any:
        for n in names:
            if n in s and s[n] is not None:
                return s[n]
        return None

    rec["ttft_mean_ms"] = pick("mean_ttft_ms", "ttft_mean")
    rec["ttft_p50_ms"] = pick("median_ttft_ms", "p50_ttft_ms")
    rec["ttft_p95_ms"] = pick("p95_ttft_ms")
    rec["ttft_p99_ms"] = pick("p99_ttft_ms")
    rec["req_throughput"] = pick("request_throughput", "throughput")
    rec["output_throughput"] = pick("output_throughput")
    rec["total_input_tokens"] = pick("total_input_tokens")
    rec["completed"] = pick("completed", "num_completed")
    return rec


def fmt(v: Any, spec: str = ".1f") -> str:
    if v is None:
        return "-"
    try:
        return format(float(v), spec)
    except Exception:
        return str(v)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: aggregate_matrix.py <results_dir>", file=sys.stderr)
        return 2
    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        return 2

    runs: list[dict[str, Any]] = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        if "_" not in sub.name:
            continue
        rec = load_run(sub)
        if rec is None:
            continue
        runs.append(rec)

    if not runs:
        print("(no runs found)")
        return 1

    # Group by (group_id) → {mode: rec}
    by_gid: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        gid = r["group_id"]
        if not gid:
            continue
        by_gid.setdefault(gid, {})[r["mode"]] = r

    def gid_sort_key(g: str) -> int:
        try:
            return int(g.lstrip("G"))
        except Exception:
            return 9999

    print("# ALL vs ALIGN Matrix Benchmark Results\n")
    print(f"Results dir: `{results_dir}`\n")

    # Detailed per-mode table
    print("## Per-group raw metrics\n")
    print("| group | prefix_len | K | num_prefixes | mode | status | ttft_p50 (ms) | ttft_p95 (ms) | ttft_p99 (ms) | req/s | completed |")
    print("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for gid in sorted(by_gid.keys(), key=gid_sort_key):
        for mode in ("align", "all"):
            r = by_gid[gid].get(mode)
            if r is None:
                print(f"| {gid} | - | - | - | {mode} | absent | - | - | - | - | - |")
                continue
            print(
                f"| {gid} | {r['prefix_len']} | {r['K']} | {r['num_prefixes']} | "
                f"{mode} | {r['status']} | "
                f"{fmt(r['ttft_p50_ms'])} | {fmt(r['ttft_p95_ms'])} | {fmt(r['ttft_p99_ms'])} | "
                f"{fmt(r['req_throughput'], '.2f')} | {fmt(r['completed'], '.0f')} |"
            )
    print()

    # Speedup table
    print("## ALL vs ALIGN speedup (TTFT p50)\n")
    print("| group | prefix_len | K | align p50 (ms) | all p50 (ms) | speedup (align/all) |")
    print("|---|---:|---:|---:|---:|---:|")
    for gid in sorted(by_gid.keys(), key=gid_sort_key):
        ra = by_gid[gid].get("align")
        rl = by_gid[gid].get("all")
        if not (ra and rl):
            continue
        plen = (ra or rl)["prefix_len"]
        k = (ra or rl)["K"]
        a_p50 = ra.get("ttft_p50_ms") if ra else None
        l_p50 = rl.get("ttft_p50_ms") if rl else None
        if a_p50 and l_p50 and float(l_p50) > 0:
            speedup = float(a_p50) / float(l_p50)
            speedup_s = f"{speedup:.2f}×"
        else:
            speedup_s = "-"
        print(
            f"| {gid} | {plen} | {k} | "
            f"{fmt(a_p50)} | {fmt(l_p50)} | {speedup_s} |"
        )
    print()

    # Failure summary
    failures = [r for r in runs if r["status"] != "ok"]
    if failures:
        print("## Failures / Missing\n")
        for r in failures:
            print(f"- {r['group_id']}_{r['mode']}: {r['status']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
