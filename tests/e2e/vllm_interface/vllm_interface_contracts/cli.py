# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Command-line interface for the vLLM PR compatibility check."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .range_analysis import (
    analyze_range,
    render_vllm_pr_summary,
)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--expect-ascend-sha", required=True)
    parser.add_argument("--index-workers", type=int, default=1)


def _range_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze-range", help="Analyze an exact vLLM PR base-to-head range.")
    _add_sources(parser)
    parser.add_argument("--old", required=True, help="vLLM PR base commit")
    parser.add_argument("--new", required=True, help="vLLM PR head commit")
    parser.add_argument("--fail-on", choices=("never", "introduced", "unresolved"), default="never")
    parser.add_argument("--analysis-workers", type=int, default=3)


def _analyze(args: argparse.Namespace) -> int:
    report = analyze_range(
        vllm_root=args.vllm_root,
        ascend_root=args.ascend_root,
        old=args.old,
        new=args.new,
        expect_ascend_sha=args.expect_ascend_sha,
        analysis_workers=args.analysis_workers,
        index_workers=args.index_workers,
    )
    print(render_vllm_pr_summary(report))
    actionable_introduced = report["summary"]["actionable_introduced_break"]
    if args.fail_on == "introduced" and actionable_introduced:
        return 1
    if args.fail_on == "unresolved" and (actionable_introduced or report["summary"]["analysis_unresolved"]):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _range_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "analyze-range":
            return _analyze(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 2
