# SPDX-License-Identifier: Apache-2.0
"""Analyze completed Flash calibration runs; never approve or overwrite a baseline."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    root = Path(root)
    reference = root / "graph-0"
    cases = read(reference / "result.json")["lengths"]
    if len(cases) < 19:
        raise ValueError("Full boundary collection required, not smoke")
    weights = read(reference / "weights.json")
    if [w["rank"] for w in weights] != list(range(4)):
        raise ValueError("Four ordered rank fingerprints required")
    settings = read(reference / "settings.json")
    runtime = read(reference / "runtime-settings.json")
    comparisons = {}
    rates, ttft, tpot = [], [], []
    run_medians = []
    for run in ("graph-0", "graph-1", "graph-2", "eager-0"):
        folder = root / run
        result = read(folder / "result.json")
        if result["status"] != "PASS" or result["lengths"] != cases:
            raise ValueError(f"Incomplete precision run: {run}")
        replays = result["replays"]
        if len(replays) != 4 or not all(n > 0 if run.startswith("graph") else n == 0 for n in replays):
            raise ValueError(f"Invalid replay evidence: {run}")
        if read(folder / "weights.json") != weights or read(folder / "runtime-settings.json") != runtime:
            raise ValueError(f"Weight/runtime drift: {run}")
        current_settings = read(folder / "settings.json")
        expected = dict(settings)
        if run.startswith("eager"):
            expected["enforce_eager"] = True
            expected.pop("compilation_config", None)
        if current_settings != expected:
            raise ValueError(f"Settings drift: {run}")
        deltas, top1 = [], []
        for length in cases:
            name = f"n{length}-cold.npy"
            a = np.load(reference / name, allow_pickle=False)
            b = np.load(folder / name, allow_pickle=False)
            if a.shape != (8, 154880) or a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
                raise ValueError(f"Invalid logits: {run}/{name}")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Non-finite logits: {run}/{name}")
            deltas.append(float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
            top1.append(bool(np.array_equal(a.argmax(-1), b.argmax(-1))))
        comparisons[run] = {"max_abs": max(deltas), "top1_equal": all(top1), "case_max_abs": dict(zip(cases, deltas))}
        if run.startswith("graph"):
            rows = read(folder / "performance.json")
            if len(rows) != 15 or sorted(r["iteration"] for r in rows) != list(range(15)):
                raise ValueError(f"Incomplete performance samples: {run}")
            values = []
            for row in rows:
                rate = row["output_tokens_s"]
                wall = row["wall_ms"]
                if not np.isfinite(wall) or wall <= 0:
                    raise ValueError("Invalid wall time")
                if not np.isfinite(rate) or rate <= 0 or not np.isclose(rate, 256000 / wall, rtol=1e-8):
                    raise ValueError("Invalid throughput")
                if len(row["requests"]) != 4:
                    raise ValueError("Missing request metrics")
                for request in row["requests"]:
                    if request["preemptions"] != 0:
                        raise ValueError("Preempted measurement")
                    for metric in ("ttft_ms", "tpot_ms"):
                        if not np.isfinite(request[metric]) or request[metric] <= 0:
                            raise ValueError("Invalid latency")
                    ttft.append(request["ttft_ms"])
                    tpot.append(request["tpot_ms"])
                values.append(rate)
            rates.extend(values)
            run_medians.append(statistics.median(values))
    manifest = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for run in ("graph-0", "graph-1", "graph-2", "eager-0")
        for p in sorted((root / run).iterdir())
        if p.is_file()
    }
    return {
        "status": "COLLECTED_NOT_APPROVED",
        "comparisons": comparisons,
        "performance": {
            "samples": len(rates),
            "median_tokens_s": statistics.median(rates),
            "run_medians_tokens_s": run_medians,
            "ttft_median_ms": statistics.median(ttft),
            "tpot_median_ms": statistics.median(tpot),
            "ttft_p95_ms": float(np.percentile(ttft, 95)),
            "tpot_p95_ms": float(np.percentile(tpot, 95)),
        },
        "sha256": manifest,
        "limitations": ["Synthetic weights", "MTP off", "Prefix cache off", "No task accuracy certification"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))
