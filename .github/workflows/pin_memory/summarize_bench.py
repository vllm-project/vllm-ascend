#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render bench/npurt/env logs as markdown for $GITHUB_STEP_SUMMARY.

The workflow pipes this into the job summary so the four-arm table and the
npurt probe results are visible directly on the GitHub run page.
"""

from pathlib import Path

ARMS = {"pageable", "torch", "native-unregistered", "native-registered"}


def parse_bench(log: Path):
    header = None
    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if line.startswith("arm,"):
            header = line.split(",")
        elif header and line:
            parts = line.split(",")
            if len(parts) == len(header) and parts[0] in ARMS:
                rows.append(parts)
    return header, rows


def render_bench(header, rows):
    if not header:
        return ["> (no bench table found in bench.log)", ""]
    out = ["### h2d/d2h bench", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def main():
    out = ["# Pin Memory Bench", ""]

    env_log = Path("env.log")
    out += ["### environment", "", "```"]
    if env_log.exists():
        out += env_log.read_text().splitlines()[:30]
    else:
        out += ["(env.log missing)"]
    out += ["```", ""]

    bench_log = Path("bench.log")
    if bench_log.exists():
        header, rows = parse_bench(bench_log)
        out += render_bench(header, rows)

    npurt_log = Path("npurt.log")
    out += ["### npurt probe", "", "```"]
    if npurt_log.exists():
        lines = [line for line in npurt_log.read_text().splitlines() if line.startswith("[") or "hasattr" in line]
        out += lines[:40] or ["(no check lines)"]
    else:
        out += ["(npurt.log missing)"]
    out += ["```"]

    # Tail of each additional experiment log, newest results last.
    for name, tag in (
        ("verify_pin.log", "verify_pin_memory --safe"),
        ("ut.log", "pin_memory unit tests"),
        ("lottery.log", "register lottery probe"),
    ):
        p = Path(name)
        if p.exists():
            out += [f"### {tag}", "", "```"]
            out += p.read_text().splitlines()[-20:]
            out += ["```", ""]

    # Training experiment results (maxseq phase and optional challenge).
    import json

    maxseq = Path("out/results_maxseq.jsonl")
    if maxseq.exists():
        out += [
            "### training results (maxseq + full rounds)",
            "",
            "| experiment | max_seq | speedup | status |",
            "|---|---|---|---|",
        ]
        for line in maxseq.read_text().splitlines():
            try:
                s = json.loads(line)
            except ValueError:
                continue
            ratio = s.get("speedup") or s.get("pin_speedup")
            out.append(f"| {s.get('experiment')} | {s.get('max_seq')} | {ratio} | {s.get('status')} |")
        out += [""]
    challenge = Path("out/results_challenge.jsonl")
    if challenge.exists():
        out += ["### challenge rounds", "", "| experiment | seq | best_round | best ratio |", "|---|---|---|---|"]
        for line in challenge.read_text().splitlines():
            try:
                s = json.loads(line)
            except ValueError:
                continue
            best = s.get("best") or {}
            ratio = best.get("speedup") or best.get("pin_speedup")
            out.append(f"| {s.get('experiment')} | {s.get('seq')} | {s.get('best_round')} | {ratio} |")
        out += [""]

    print("\n".join(out))


if __name__ == "__main__":
    main()
