#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate on run_maxseq.py output (out/results_maxseq.jsonl).

Emits ::error:: / ::notice:: annotations and exits non-zero when any planned
experiment row is missing or failed, so a broken training run fails the CI
step instead of passing green. 'no-seq-fits' rows are accepted (they are a
valid result for that card) and reported as notices.
"""

import json
import sys
from pathlib import Path

EXPECTED = {
    "delock_maxseq_Qwen3-4B",
    "delock_maxseq_Qwen3.5-4B",
    "act_maxseq_Qwen3-4B",
    "act_maxseq_Qwen3.5-4B",
    "sao_maxseq_Qwen3-4B",
    "sao_maxseq_Qwen3.5-4B",
}


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "out/results_maxseq.jsonl")
    if not path.exists():
        print(f"::error::training results file {path} not found: run_maxseq.py did not complete")
        sys.exit(1)

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"::error::unparsable results line: {line[:120]}")
            sys.exit(1)

    seen = {}
    for r in rows:
        seen[r.get("experiment", "?")] = r

    problems = 0
    for exp in sorted(EXPECTED - set(seen)):
        print(f"::error::missing experiment row: {exp}")
        problems += 1

    for exp, r in sorted(seen.items()):
        status = r.get("status")
        if status == "ok":
            ratio = r.get("speedup") or r.get("pin_speedup")
            print(f"[ok] {exp}: max_seq={r.get('max_seq')} ratio={ratio}")
        elif status == "no-seq-fits":
            print(f"::notice::{exp}: no seq fits on this card")
        else:
            print(f"::error::{exp}: status={status}")
            problems += 1

    if problems:
        print(f"::error::training gate FAILED with {problems} problem(s)")
        sys.exit(1)
    print(f"training gate PASSED ({len(seen)} rows)")


if __name__ == "__main__":
    main()
