#!/usr/bin/env python3
"""Summarize throughput and exact-text accuracy for DSpark TP experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


TPS = (1, 2, 4, 8, 16)
CURRENT_CASES = ("target", "fixed", "adaptive", "physical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", required=True)
    parser.add_argument("--base-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tps", nargs="+", type=int, default=TPS)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def throughput(case_dir: Path) -> tuple[list[float], float]:
    values = [read_json(path)["output_throughput"] for path in sorted(case_dir.glob("benchmark-*.json"))]
    if not values:
        return [], float("nan")
    return values, statistics.median(values)


def texts(case_dir: Path) -> list[str] | None:
    path = case_dir / "accuracy.json"
    if not path.exists():
        return None
    return [item["text"] for item in read_json(path)]


def digest(values: list[str] | None) -> str | None:
    if values is None:
        return None
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def gain(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


def main() -> int:
    args = parse_args()
    current_root = Path(args.current_root)
    base_root = Path(args.base_root) if args.base_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    accuracy_rows: list[dict] = []
    for tp in args.tps:
        target_dir = current_root / f"tp{tp}" / "target"
        target_values, target_median = throughput(target_dir)
        target_texts = texts(target_dir)
        for case in CURRENT_CASES:
            case_dir = current_root / f"tp{tp}" / case
            values, median = throughput(case_dir)
            case_texts = texts(case_dir)
            rows.append(
                {
                    "source": "current",
                    "tp": tp,
                    "case": case,
                    "runs": len(values),
                    "throughputs": values,
                    "median_output_throughput": median,
                    "gain_vs_current_target_pct": gain(median, target_median),
                    "accuracy_matches_current_target": case_texts == target_texts,
                    "accuracy_sha256": digest(case_texts),
                }
            )
            accuracy_rows.append(
                {
                    "tp": tp,
                    "comparison": f"current_{case}_vs_current_target",
                    "matches": case_texts == target_texts,
                }
            )

        if base_root is not None:
            for case in ("target", "adaptive"):
                base_dir = base_root / f"tp{tp}" / case
                base_values, base_median = throughput(base_dir)
                current_values, current_median = throughput(current_root / f"tp{tp}" / case)
                base_texts = texts(base_dir)
                current_texts = texts(current_root / f"tp{tp}" / case)
                rows.append(
                    {
                        "source": "base",
                        "tp": tp,
                        "case": case,
                        "runs": len(base_values),
                        "throughputs": base_values,
                        "median_output_throughput": base_median,
                        "gain_vs_current_target_pct": gain(base_median, target_median),
                        "current_vs_base_pct": gain(current_median, base_median),
                        "accuracy_matches_current": base_texts == current_texts,
                        "accuracy_sha256": digest(base_texts),
                    }
                )
                accuracy_rows.append(
                    {
                        "tp": tp,
                        "comparison": f"current_{case}_vs_base_{case}",
                        "matches": current_texts == base_texts,
                    }
                )

    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "accuracy-summary.json").write_text(
        json.dumps(accuracy_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row if key != "throughputs"})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
