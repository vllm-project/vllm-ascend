#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License")
#
# you may not use this file except in compliance with the License
#
# You may obtain a copy of the License at
#
# <http://www.apache.org/licenses/LICENSE-2.0>
#
# Unless required by applicable law or agreed to in writing, software
#
# distributed under the License is distributed on an "AS IS" BASIS
#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
#
# See the License for the specific language governing permissions and
#
# limitations under the License
#
# This file is a part of the vllm-ascend project
#

"""Generate a compact PR comment from diff-cover and slipcover output.

diff-cover produces a markdown report containing a Summary section:

    ## Summary

    - **Total**: 191 lines
    - **Missing**: 85 lines
    - **Coverage**: 55%

With --input-json, the overall coverage from a slipcover coverage.json
supplies the Total Coverage column; the diff-cover summary supplies the
Diff Coverage column. The output is a 4x3 markdown table:

    |              | Total Coverage   | Diff Coverage |
    | ------------ | --------------   | ------------- |
    | **Total**    | 145055 lines     | 191 lines     |
    | **Missing**  | 41592 lines      | 85 lines      |
    | **Coverage** | 71% | <mark>~~***55%***~~<mark> |

Coverage cells render as highlighted strikethrough-bold-italic when below
the threshold (Total < 70%, Diff < 80%), plain when at/above.
"""

import argparse
import json
import sys
from pathlib import Path

import regex as re


def _parse_bullet(line: str) -> tuple:
    """Extract (label, value_str, percent_or_None) from a bullet like
    '- **Total**: 191 lines' or '- **Coverage**: 55%'."""
    m = re.match(r"- \*\*(\w+)\*\*:\s*(.+)", line.strip())
    if not m:
        return None
    label = m.group(1)
    value = m.group(2).strip()
    pct = None
    pm = re.search(r"(\d+(?:\.\d+)?)%", value)
    if pm:
        pct = float(pm.group(1))
    return label, value, pct


def extract_diff_summary(markdown: str) -> dict | None:
    """Extract Total/Missing/Coverage from the diff-cover Summary section.

    Returns {"total": str, "missing": str, "coverage_pct": float} or None.
    """
    result: dict = {}
    in_summary = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## Summary"):
            in_summary = True
            continue
        if in_summary:
            if stripped.startswith("#"):
                break
            if stripped.startswith("- **"):
                parsed = _parse_bullet(stripped)
                if parsed:
                    key = parsed[0].lower()
                    if key in ("total", "missing", "coverage"):
                        result[key] = parsed[1]
                        if parsed[2] is not None:
                            result["coverage_pct"] = parsed[2]
    if "total" not in result or "missing" not in result or "coverage_pct" not in result:
        return None
    return result


def extract_total_coverage(json_text: str) -> dict | None:
    """Extract Total/Missing/Coverage from slipcover coverage.json.

    Returns {"total": str, "missing": str, "coverage_pct": float} or None.
    """
    try:
        summary = json.loads(json_text)["summary"]
        covered = int(summary["covered_lines"])
        missing = int(summary["missing_lines"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    percent = summary.get("percent_covered_display")
    if percent is None:
        percent = round(float(summary["percent_covered"]))
    if isinstance(percent, float) and percent.is_integer():
        percent = int(percent)
    return {
        "total": f"{covered + missing} lines",
        "missing": f"{missing} lines",
        "coverage_pct": float(percent),
    }


def _fmt_coverage(pct: float, threshold: float) -> str:
    """Format a coverage cell: highlight below threshold, plain at/above."""
    pct_str = f"{pct}%"
    if isinstance(pct, float) and pct.is_integer():
        pct_str = f"{int(pct)}%"
    if pct < threshold:
        return f"<mark>~~***{pct_str}***~~<mark>"
    return pct_str


def build_table(total: dict, diff: dict | None, summary_url: str = "") -> str:
    """Render the 4x3 markdown table with conditional formatting."""
    total_cov = total["coverage_pct"]
    total_cell = _fmt_coverage(total_cov, 70)

    if diff:
        diff_cov = diff["coverage_pct"]
        diff_cell = _fmt_coverage(diff_cov, 80)
    else:
        diff_cell = "N/A"

    diff_total = diff["total"] if diff else "N/A"
    diff_missing = diff["missing"] if diff else "N/A"

    out = (
        "## Coverage\n"
        "|              | Total Coverage  | Diff Coverage |\n"
        "| ------------ | --------------- | ------------- |\n"
        f"| **Total**    | {total['total']}         | {diff_total} |\n"
        f"| **Missing**  | {total['missing']}     | {diff_missing} |\n"
        f"| **Coverage** | {total_cell}       | {diff_cell} |\n"
    )
    if summary_url:
        out += f"\nMore info see: [Summary]({summary_url})\n"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Path to coverage.json produced by slipcover")
    parser.add_argument("--input-diff", required=True, help="Path to diff_coverage.md")
    parser.add_argument("--output", required=True, help="Path to write coverage_comment.md")
    parser.add_argument("--summary-url", default="", help="URL to the cpu-ut job execution page")
    args = parser.parse_args()

    json_path = Path(args.input_json)
    if not json_path.exists():
        print(f"::warning::input json file {json_path} not found; no coverage comment generated", file=sys.stderr)
        return 0
    total = extract_total_coverage(json_path.read_text(encoding="utf-8"))
    if not total:
        print(f"::warning::no top-level summary in {json_path}; no coverage comment generated", file=sys.stderr)
        return 0

    diff_path = Path(args.input_diff)
    diff = None
    if diff_path.exists():
        diff = extract_diff_summary(diff_path.read_text(encoding="utf-8"))
    else:
        print(f"::warning::input file {diff_path} not found; diff coverage will show N/A", file=sys.stderr)

    comment = build_table(total, diff, args.summary_url)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(comment, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
