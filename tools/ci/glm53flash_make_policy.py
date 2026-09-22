# SPDX-License-Identifier: Apache-2.0
"""Create an UNAPPROVED, host-specific candidate policy from measured data."""

import argparse
import json
from pathlib import Path


def candidate_policy(summary, scope="UNSPECIFIED"):
    comparisons = summary["comparisons"]
    if not all(value["top1_equal"] for value in comparisons.values()):
        raise ValueError("Top-1 discrepancy needs investigation before proposing a precision gate")
    graph_error = max(comparisons[f"graph-{i}"]["max_abs"] for i in range(3))
    perf = summary["performance"]
    return {
        "schema_version": 1,
        "approved": False,
        "approved_by": None,
        "baseline_correctness_validated": False,
        "blocking_findings": [
            "Synthetic sliced weights do not certify real-model task accuracy.",
            "MTP reject-path execution is covered, but the accepted-token path is not covered.",
            "Single-image service smoke and the real-weight single-node TP16 nightly remain pending.",
            "A second independent collection must pass the candidate policy before human approval.",
        ],
        "precision_atol": graph_error * 1.25,
        "eager_graph_atol": comparisons["eager-0"]["max_abs"] * 1.25,
        "minimum_tokens_s": min(perf["run_medians_tokens_s"]) * 0.90,
        "maximum_ttft_p95_ms": perf["ttft_p95_ms"] * 1.15,
        "maximum_tpot_p95_ms": perf["tpot_p95_ms"] * 1.15,
        "baseline_sha256": summary["sha256"],
        "scope": scope,
        "notes": [
            "Candidate only: margins are engineering choices, not statistical guarantees.",
            "Exact graph equality is proposed only when observed repeated error is zero.",
            "Requires exclusive matching runner; not portable across images or hardware.",
            "Synthetic weights; no full-model task accuracy claim.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", required=True, help="Exact host, devices, image/runtime and model scope")
    args = parser.parse_args()
    policy = candidate_policy(json.loads(args.summary.read_text()), args.scope)
    with args.output.open("x") as output:
        json.dump(policy, output, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in policy.items() if k != "baseline_sha256"}, indent=2))
