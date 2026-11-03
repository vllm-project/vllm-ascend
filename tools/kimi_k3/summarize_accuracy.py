# SPDX-License-Identifier: Apache-2.0
"""Summarize completed AISBench result files, one file per independent round."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths):
    rounds = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if len(result["details"]) != 500:
            raise ValueError(f"Expected 500 evaluation details: {path}")
        accuracy = float(result["accuracy"])
        if not 0 <= accuracy <= 100:
            raise ValueError(f"Invalid accuracy: {path}")
        rounds.append({"path": str(path), "count": 500, "accuracy_percent": accuracy})
    if len(rounds) != 3:
        raise ValueError("Expected three completed independent rounds")
    values = [row["accuracy_percent"] for row in rounds]
    return {
        "rounds": rounds,
        "mean_accuracy_percent": statistics.mean(values),
        "population_std_percentage_points": statistics.pstdev(values),
        "sample_std_percentage_points": statistics.stdev(values),
        "range_percentage_points": max(values) - min(values),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.results), indent=2) + "\n", encoding="utf-8")
