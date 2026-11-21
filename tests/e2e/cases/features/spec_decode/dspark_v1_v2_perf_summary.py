#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Summarize a ``dspark_v1_v2_perf.sh`` sweep into Markdown tables.

Read every ``*.json`` written by ``vllm bench serve``. Each file must carry an
``ascend_run_meta`` block added by the sweep script. V1 and V2 runs are paired
by (num_speculative_tokens, max_num_seqs, enforce_eager_draft) for like-for-like
comparison.

Usage::

    python dspark_v1_v2_perf_summary.py ./perf_results --out perf_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (bench-serve key, column label, "higher is better")
METRICS: list[tuple[str, str, bool]] = [
    ("output_throughput", "Output throughput (tok/s)", True),
    ("total_token_throughput", "Total token throughput (tok/s)", True),
    ("mean_ttft_ms", "Mean TTFT (ms)", False),
    ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ("mean_tpot_ms", "Mean TPOT (ms)", False),
    ("mean_e2el_ms", "Mean end-to-end latency (ms)", False),
    ("p99_e2el_ms", "P99 end-to-end latency (ms)", False),
]


def _load_runs(result_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(result_dir.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON: {path}")
                continue
        meta = data.get("ascend_run_meta")
        if not meta:
            print(f"[warn] skipping {path}: no ascend_run_meta block")
            continue
        runs.append({"path": path, "meta": meta, "data": data})
    return runs


def _key(meta: dict[str, Any]) -> tuple[int, int, bool]:
    return (
        meta["num_speculative_tokens"],
        meta["max_num_seqs"],
        meta["enforce_eager_draft"],
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _ratio(v1: Any, v2: Any, higher_is_better: bool, regression_threshold: float) -> str:
    """Format V2 relative to V1 with an unambiguous direction."""
    if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)) or not v1:
        return "-"
    ratio = v2 / v1
    if higher_is_better:
        delta_pct = (ratio - 1.0) * 100
    else:
        # Lower is better, so a smaller V2 value is an improvement.
        delta_pct = (1.0 - ratio) * 100
    marker = "PASS" if delta_pct >= -regression_threshold else "REGRESSION"
    return f"{ratio:.3f} ({delta_pct:+.1f}% {marker})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        help="Flag a data point when V2 is worse than V1 by more than this percentage.",
    )
    args = parser.parse_args()

    runs = _load_runs(args.result_dir)
    if not runs:
        raise SystemExit(f"no usable result JSON under {args.result_dir}")

    paired: dict[tuple[int, int, bool], dict[str, dict[str, Any]]] = {}
    for run in runs:
        paired.setdefault(_key(run["meta"]), {})[run["meta"]["model_runner"]] = run

    lines: list[str] = ["# DSpark performance: Model Runner V2 vs V1\n"]
    lines.append(
        "Each section pairs V1 and V2 for one "
        "(num_speculative_tokens, batch, draft graph mode) configuration. "
        "The ratio is V2/V1; the percentage is normalized so positive is better. "
        f"A degradation beyond {args.regression_threshold}% is marked REGRESSION.\n"
    )

    regressions: list[str] = []
    for key in sorted(paired):
        spec_tokens, batch_size, eager = key
        arms = paired[key]
        v1, v2 = arms.get("v1"), arms.get("v2")
        lines.append(f"## num_speculative_tokens={spec_tokens}, batch={batch_size}, draft_eager={eager}\n")

        if not v1 or not v2:
            present = sorted(arms)
            lines.append(f"> This configuration has no paired arm (present: {present}); comparison is unavailable.\n")
            continue

        measurements = [
            (
                label,
                higher_is_better,
                v1["data"].get(metric_key),
                v2["data"].get(metric_key),
            )
            for metric_key, label, higher_is_better in METRICS
        ] + [
            (
                "Peak HBM (MB)",
                False,
                v1["meta"].get("peak_hbm_mb"),
                v2["meta"].get("peak_hbm_mb"),
            )
        ]

        lines.append("| Metric | V1 | V2 | V2/V1 |")
        lines.append("| --- | --- | --- | --- |")
        for label, higher_is_better, value_v1, value_v2 in measurements:
            lines.append(
                f"| {label} | {_fmt(value_v1)} | {_fmt(value_v2)} | "
                f"{_ratio(value_v1, value_v2, higher_is_better, args.regression_threshold)} |"
            )
            if isinstance(value_v1, (int, float)) and isinstance(value_v2, (int, float)) and value_v1:
                ratio = value_v2 / value_v1
                delta_pct = (ratio - 1.0) * 100 if higher_is_better else (1.0 - ratio) * 100
                if delta_pct < -args.regression_threshold:
                    regressions.append(f"spec={spec_tokens} bs={batch_size} eager={eager}: {label} {delta_pct:+.1f}%")
        lines.append("")

    lines.append("## Regression verdict\n")
    if regressions:
        lines.append(f"Found {len(regressions)} measurements worse than the {args.regression_threshold}% threshold:\n")
        lines.extend(f"- {item}" for item in regressions)
        lines.append(
            "\nConfirm that each result exceeds measurement noise before filing a follow-up linked to issue #13745.\n"
        )
    else:
        lines.append(f"- No paired metric degraded by more than {args.regression_threshold}%.\n")

    report = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[summary] wrote {args.out} ({len(paired)} combinations, {len(regressions)} regressions)")
    else:
        print(report)


if __name__ == "__main__":
    main()
