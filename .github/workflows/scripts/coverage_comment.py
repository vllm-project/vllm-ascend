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

"""Generate a compact PR comment from diff-cover output.

diff-cover produces a markdown report like:

    # Diff Coverage
    ## Diff: origin/main...HEAD, staged and unstaged changes

    - vllm_ascend/foo.py (50.0%): Missing lines 10-12

    ## Summary

    - **Total**: 20 lines
    - **Missing**: 10 lines
    - **Coverage**: 50%

    ## vllm_ascend/foo.py
    ...

This script extracts the H1 title and the three summary bullet points
(Total, Missing, Coverage) and writes them to the output file as a
compact PR comment.
"""

import argparse
import sys
from pathlib import Path


def extract_lines(markdown: str) -> list[str]:
    """Extract the H1 title and the Summary section bullet points.

    Returns a list of lines in the order: title, total, missing, coverage.
    """
    title = None
    summary_bullets = []

    in_summary = False
    for line in markdown.splitlines():
        stripped = line.strip()

        # Capture the top-level title (# Diff Coverage)
        if title is None and stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped
            continue

        # Enter the Summary section
        if stripped.startswith("## Summary"):
            in_summary = True
            continue

        # Once in Summary, collect bullet lines until next heading
        if in_summary:
            if stripped.startswith("#"):
                break
            if stripped.startswith("- **"):
                summary_bullets.append(stripped)

    return [line for line in (title,) + tuple(summary_bullets) if line is not None]


def build_comment(lines: list[str], summary_url: str = "") -> str:
    """Render the extracted lines as a markdown PR comment."""
    if not lines:
        return "## Diff Coverage\n\nNo coverage data found.\n"
    out = "\n".join(lines) + "\n"
    if summary_url:
        out += f"\nMore info see: [Summary]({summary_url})\n"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to diff_coverage.md")
    parser.add_argument("--output", required=True, help="Path to write coverage_comment.md")
    parser.add_argument("--summary-url", default="", help="URL to the cpu-ut job execution page")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"::warning::input file {in_path} not found; no coverage comment generated", file=sys.stderr)
        return 0

    markdown = in_path.read_text(encoding="utf-8")
    lines = extract_lines(markdown)
    comment = build_comment(lines, args.summary_url)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(comment, encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
