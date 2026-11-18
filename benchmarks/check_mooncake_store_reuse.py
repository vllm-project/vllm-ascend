# SPDX-License-Identifier: Apache-2.0
"""Check paired Mooncake Store reuse measurements against the recorded gates.

This consumes measurements; it does not start an engine or collect new timings.
Run: python benchmarks/check_mooncake_store_reuse.py measurements.json
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def summarize(measurements: dict) -> dict:
    pairs = measurements["pairs"]
    if len(pairs) != 10:
        raise ValueError("The acceptance protocol requires ten A/B pairs")

    statistics = {}
    valid_timings = True
    for phase in ("source", "restore"):
        for metric in ("ttft_seconds", "tpot_seconds", "wall_seconds"):
            values = {arm: [pair[arm][phase][metric] for pair in pairs] for arm in ("A", "B")}
            valid_timings &= all(math.isfinite(x) and x > 0 for rows in values.values() for x in rows)
            statistics[f"{phase}_{metric}"] = {
                arm: {"median": median(rows), "min": min(rows), "max": max(rows)} for arm, rows in values.items()
            }
            statistics[f"{phase}_{metric}"].update(
                paired_median=median(b - a for a, b in zip(values["A"], values["B"])),
                B_slower_pairs=sum(b > a for a, b in zip(values["A"], values["B"])),
            )

    restores = [pair[arm]["restore"] for pair in pairs for arm in ("A", "B")]
    sources = [pair[arm]["source"] for pair in pairs for arm in ("A", "B")]
    checks = {
        "positive_finite_timings": valid_timings,
        "matched_workload": all(
            pair["A"][phase]["prompt_tokens"] == pair["B"][phase]["prompt_tokens"]
            and pair["A"][phase]["generated_tokens"] == pair["B"][phase]["generated_tokens"]
            for pair in pairs
            for phase in ("source", "restore")
        ),
        "fresh_sources": all(
            row["evidence"]["local"] == row["evidence"]["external"] == row["evidence"]["get_calls"] == 0
            for row in sources
        ),
        "external_restores": all(
            row["evidence"]["local"] == 0
            and row["evidence"]["external"] > 0
            and row["evidence"]["get_calls"] > 0
            and row["evidence"]["get_bytes"] > 0
            and row["evidence"]["get_failed"] == 0
            for row in restores
        ),
        "decode_checkpoint_consumed": all(
            pair["A"]["restore"]["evidence"]["external"]
            <= pair["A"]["source"]["prompt_tokens"]
            < pair["B"]["restore"]["evidence"]["external"]
            < pair["B"]["restore"]["prompt_tokens"]
            for pair in pairs
        ),
        "mtp_acceptance": all(row["mtp_accepted"] > 0 for row in sources + restores),
        "restore_ttft_improves": statistics["restore_ttft_seconds"]["paired_median"] < 0,
        "source_tpot_gate": statistics["source_tpot_seconds"]["B_slower_pairs"] < 9,
        "restore_tpot_gate": statistics["restore_tpot_seconds"]["B_slower_pairs"] < 9,
    }
    return {"passed": all(checks.values()), "checks": checks, "pairs": len(pairs), "statistics": statistics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing ten measured A/B pairs")
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
