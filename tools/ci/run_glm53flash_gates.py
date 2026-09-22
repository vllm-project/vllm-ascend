# SPDX-License-Identifier: Apache-2.0
"""Fail-closed offline Flash gates over independently collected calibration artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.ci.glm53flash_analyze import analyze, read
from tools.ci.glm53flash_precision import check_logits


def evaluate(baseline, current, policy, enforce=False, environment=None):
    if not environment or environment.get("validated") is not True or environment.get("exclusive") is not True:
        raise ValueError("Trusted runner must validate exclusive calibrated environment")
    if environment.get("scope") != policy["scope"] or not environment.get("checked_by"):
        raise ValueError("Environment scope mismatch")
    baseline, current = Path(baseline).resolve(), Path(current).resolve()
    if baseline == current or baseline in current.parents or current in baseline.parents:
        raise ValueError("Baseline and current must be independent directories")
    if enforce and (policy.get("approved") is not True or not policy.get("approved_by")):
        raise ValueError("Policy requires approval before enforcement")
    if enforce and policy.get("baseline_correctness_validated") is not True:
        raise ValueError("Baseline correctness has not been validated")
    for key in ("precision_atol", "eager_graph_atol", "minimum_tokens_s", "maximum_ttft_p95_ms", "maximum_tpot_p95_ms"):
        if not isinstance(policy[key], (int, float)) or not math.isfinite(policy[key]) or policy[key] < 0:
            raise ValueError(f"Invalid threshold: {key}")
    if policy["minimum_tokens_s"] == 0:
        raise ValueError("Throughput threshold must be positive")
    reference = analyze(baseline)
    if reference["sha256"] != policy["baseline_sha256"]:
        raise ValueError("Baseline files changed")
    candidate = analyze(current)
    for name in ("weights.json", "settings.json", "runtime-settings.json", "path-evidence.json"):
        if read(baseline / "graph-0" / name) != read(current / "graph-0" / name):
            raise ValueError(f"Baseline/current metadata mismatch: {name}")
    results = {}
    for run in ("graph-0", "graph-1", "graph-2"):
        for length in read(baseline / "graph-0" / "result.json")["lengths"]:
            name = f"n{length}-cold.npy"
            results[f"{run}/{name}"] = check_logits(
                np.load(baseline / "graph-0" / name, allow_pickle=False),
                np.load(current / run / name, allow_pickle=False),
                policy["precision_atol"],
            )
    eager = candidate["comparisons"]["eager-0"]
    precision_ok = all(v["status"] == "PASS" for v in results.values())
    precision_ok = precision_ok and eager["top1_equal"] and eager["max_abs"] <= policy["eager_graph_atol"]
    performance_ok = candidate["performance"]["median_tokens_s"] >= policy["minimum_tokens_s"]
    performance_ok = performance_ok and candidate["performance"]["ttft_p95_ms"] <= policy["maximum_ttft_p95_ms"]
    performance_ok = performance_ok and candidate["performance"]["tpot_p95_ms"] <= policy["maximum_tpot_p95_ms"]
    passed = precision_ok and performance_ok
    return {
        "status": "PASS" if passed else ("FAIL" if enforce else "WARN"),
        "exit_code": 0 if passed or not enforce else 1,
        "enforcement_enabled": enforce,
        "baseline_correctness_validated": policy.get("baseline_correctness_validated") is True,
        "precision": {"passed": precision_ok, "cases": results, "eager_graph": eager},
        "performance": {"passed": performance_ok, **candidate["performance"]},
        "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "current", "policy", "environment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(args.baseline, args.current, read(args.policy), args.enforce, read(args.environment))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {"status": "ERROR", "exit_code": 2, "reason": str(error)}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(result["status"])
    raise SystemExit(result["exit_code"])
