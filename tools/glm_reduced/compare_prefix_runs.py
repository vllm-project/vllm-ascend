# SPDX-License-Identifier: Apache-2.0
"""Compare collected prefix runs once, without generating or adjusting a baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prefix_probe import compare_arrays


def compare_runs(reference: Path, candidate: Path) -> dict:
    """Require identical workload and execution settings, report every rank/step."""
    if reference.resolve() == candidate.resolve():
        raise ValueError("expected two separate collected runs")
    left = json.loads((reference / "run.json").read_text())
    right = json.loads((candidate / "run.json").read_text())
    if left.get("role") != "reference" or right.get("role") != "candidate":
        raise ValueError("expected full reference and reduced candidate roles")
    for metadata in (left, right):
        if metadata["status"] != "COLLECTED_NOT_COMPARED":
            raise ValueError("run is incomplete")
    for key in ("engine", "requests", "continuation", "runtime", "scope", "probe_sha256", "environment"):
        if left[key] != right[key]:
            raise ValueError(f"run mismatch: {key}")
    results = []
    for item in left["requests"]:
        for rank in range(left["engine"]["tensor_parallel_size"]):
            for step in range(len(left["continuation"])):
                name = Path(item["id"]) / f"rank-{rank}" / f"step-{step:03d}.npz"
                with (
                    np.load(reference / name, allow_pickle=False) as a,
                    np.load(candidate / name, allow_pickle=False) as b,
                ):
                    if set(a.files) != set(b.files) or not {"hidden", "normalized", "position"} <= set(a.files):
                        raise ValueError(f"missing or differing capture fields: {name}")
                    if not np.array_equal(a["position"], b["position"]):
                        raise ValueError(f"different positions: {name}")
                    for key in ("hidden", "normalized", "logits"):
                        if key not in a:
                            continue  # Non-output TP ranks may return no logits.
                        metrics = compare_arrays(a[key], b[key])
                        results.append({"capture": str(name), "tensor": key, **metrics})
                    actual_norm = candidate / name.parent / f"actual-norm-{step:03d}.npz"
                    with np.load(actual_norm, allow_pickle=False) as normal:
                        results.append(
                            {
                                "capture": str(name),
                                "tensor": "candidate_early_norm_vs_actual_norm",
                                **compare_arrays(b["normalized"], normal["normalized"]),
                            }
                        )
                    if "logits" in b:
                        final = candidate / name.parent / f"final-{step:03d}.npz"
                        with np.load(final, allow_pickle=False) as normal:
                            results.append(
                                {
                                    "capture": str(name),
                                    "tensor": "candidate_early_head_vs_actual_final",
                                    **compare_arrays(b["logits"], normal["logits"]),
                                }
                            )
    expected_logits = len(left["requests"]) * len(left["continuation"])
    if sum(record["tensor"] == "logits" for record in results) < expected_logits:
        raise ValueError("not enough full-vocabulary logits captures")
    return {
        "status": "UNASSESSED",
        "reason": "Diagnostic measurements only; no calibrated numerical tolerances or accepted baseline.",
        "all_arrays_exact": all(record["max_abs_error"] == 0 for record in results),
        "reference": str(reference),
        "candidate": str(candidate),
        "results": results,
    }


def main():
    """Write measured differences; do not retry, tune tolerances, or accept a baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.reference, args.candidate)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
