# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize forward timing logs; keep workers, phases and actual BS separate."""

import argparse
import json
from pathlib import Path

import regex as re


def summarize(paths, tp_rank=0):
    groups = {}
    required = {"window", "rank", "role", "phase", "bs", "count", "avg_ms", "min_ms", "max_ms", "dropped", "errors"}
    for path in paths:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if "Forward timing:" not in line:
                    continue
                fields = dict(re.findall(r"(\w+)=([^\s]+)", line.split("Forward timing:", 1)[1]))
                if required - fields.keys():
                    raise ValueError(f"{path}:{line_number}: incomplete forward timing row")
                # Legacy plugin logs put DP/TP in the process prefix.
                dp = re.search(r"Worker_DP(\d+)", line)
                tp = re.search(r"_TP(\d+)", line)
                actual_tp = int(fields.get("tp_rank", tp.group(1) if tp else fields["rank"]))
                if tp_rank is not None and actual_tp != tp_rank:
                    continue
                pid = re.search(r"pid=(\d+)", line)
                key = (
                    str(path),
                    pid.group(1) if pid else "unknown",
                    int(fields.get("dp_rank", dp.group(1) if dp else 0)),
                    int(fields.get("pp_rank", 0)),
                    actual_tp,
                    fields["role"],
                    fields["phase"],
                    int(fields["bs"]),
                )
                group = groups.setdefault(
                    key,
                    {
                        "source": key[0],
                        "pid": key[1],
                        "dp_rank": key[2],
                        "pp_rank": key[3],
                        "tp_rank": key[4],
                        "role": key[5],
                        "phase": key[6],
                        "bs": key[7],
                        "samples": 0,
                        "total_ms": 0.0,
                        "min_ms": None,
                        "max_ms": 0.0,
                        "dropped": 0,
                        "errors": 0,
                    },
                )
                count = int(fields["count"])
                group["samples"] += count
                group["total_ms"] += count * float(fields["avg_ms"])
                group["dropped"] += int(fields["dropped"])
                group["errors"] += int(fields["errors"])
                if count:
                    minimum = float(fields["min_ms"])
                    group["min_ms"] = minimum if group["min_ms"] is None else min(group["min_ms"], minimum)
                    group["max_ms"] = max(group["max_ms"], float(fields["max_ms"]))
    for group in groups.values():
        total_ms = group.pop("total_ms")
        group["avg_ms"] = total_ms / group["samples"] if group["samples"] else None
    return [groups[key] for key in sorted(groups)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--tp-rank", type=int, default=0, help="representative TP rank (default: 0)")
    args = parser.parse_args()
    try:
        rows = summarize(args.logs, args.tp_rank)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("no matching forward timing rows")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
