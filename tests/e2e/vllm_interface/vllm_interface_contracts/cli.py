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
"""Command-line interface for the shared interface-contract engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import generator
from .analysis_plans import (
    MAIN2MAIN_SCENARIO,
    SCENARIOS,
    resolve_analysis_plan,
)
from .range_analysis import (
    GitSnapshot,
    analyze_range,
    git_head,
    validate_current_contracts,
    write_reports,
)


def _named_values(values: list[str], option: str, *, paths: bool = False) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects PACKAGE=VALUE: {value}")
        name, raw = value.split("=", 1)
        if not name or not raw or name in result:
            raise ValueError(f"invalid or duplicate {option}: {value}")
        result[name] = Path(raw) if paths else raw
    return result


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--expect-ascend-sha", required=True)
    parser.add_argument("--external-root", action="append", default=[], metavar="PACKAGE=PATH")
    parser.add_argument("--expect-external-sha", action="append", default=[], metavar="PACKAGE=SHA")
    parser.add_argument("--downstream-index-cache-dir", type=Path)
    parser.add_argument("--upstream-file-index-cache-dir", type=Path)
    parser.add_argument("--index-workers", type=int, default=1)


def _range_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze-range", help="Analyze an exact old-to-new vLLM range.")
    _add_sources(parser)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default=MAIN2MAIN_SCENARIO)
    parser.add_argument("--profile", choices=("exact-contracts", "expanded"), default="exact-contracts")
    parser.add_argument("--fail-on", choices=("never", "introduced", "unresolved"), default="never")
    parser.add_argument("--analysis-workers", type=int, default=3)


def _validate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Regenerate the live dependency graph and validate exact call/return contracts.",
    )
    _add_sources(parser)
    parser.add_argument("--scenario", choices=SCENARIOS, default=MAIN2MAIN_SCENARIO)
    parser.add_argument("--output", type=Path)


def _validate(args: argparse.Namespace) -> int:
    plan = resolve_analysis_plan(args.scenario)
    vllm_sha = git_head(args.vllm_root)
    ascend_sha = git_head(args.ascend_root)
    expected_ascend = generator._git_head(args.ascend_root)
    generator._verify_sha("vllm-ascend", expected_ascend, args.expect_ascend_sha)
    external_roots = _named_values(args.external_root, "--external-root", paths=True)
    external_shas = _named_values(args.expect_external_sha, "--expect-external-sha")
    if set(external_roots) != set(external_shas):
        raise ValueError("external roots and SHAs must name the same packages")
    for package, root in external_roots.items():
        generator._verify_sha(
            f"external {package}",
            generator._git_head(root),
            str(external_shas[package]),
        )
    engine = generator.InterfaceBoundaryGenerator(
        args.vllm_root,
        args.ascend_root,
        external_roots,
        source_versions={"vllm": vllm_sha, "vllm_ascend": ascend_sha, **external_shas},
        downstream_index_cache_dir=args.downstream_index_cache_dir,
        upstream_file_index_cache_dir=args.upstream_file_index_cache_dir,
        index_workers=args.index_workers,
    )
    relations, findings = engine.generate(plan)
    visible_generator_findings = findings if plan.include_generator_findings else []
    direct_calls, contract_findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(args.vllm_root, vllm_sha),
        plan,
    )
    statuses = Counter(item.status for item in visible_generator_findings)
    contract_statuses = Counter(str(item["status"]) for item in contract_findings)
    payload = {
        "inputs": {
            "vllm_sha": vllm_sha,
            "vllm_ascend_sha": ascend_sha,
            "generator_version": generator.GENERATOR_VERSION,
            "scenario": plan.scenario,
            "analysis_plan": plan.as_dict(),
            "relation_generation_timings_seconds": engine.phase_timings,
        },
        "summary": {
            "relations": sum(
                relation.upstream_package == "vllm" and relation.relation in plan.relation_types
                for relation in relations
            ),
            "relations_collected": len(relations),
            "direct_call_dependencies": len(direct_calls),
            "findings": len(visible_generator_findings) + len(contract_findings),
            "generator_findings": len(visible_generator_findings),
            "contract_findings": len(contract_findings),
            "contract_risks": contract_statuses["risk"],
            "contract_reviews": contract_statuses["review"],
            "generator_issues": sum(item.generator_issue for item in visible_generator_findings),
            "by_status": dict(sorted(statuses.items())),
        },
        "contract_findings": contract_findings,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    external_roots = _named_values(args.external_root, "--external-root", paths=True)
    external_shas = _named_values(args.expect_external_sha, "--expect-external-sha")
    report = analyze_range(
        vllm_root=args.vllm_root,
        ascend_root=args.ascend_root,
        old=args.old,
        new=args.new,
        expect_ascend_sha=args.expect_ascend_sha,
        external_roots=external_roots,
        external_shas=external_shas,
        profile=args.profile,
        scenario=args.scenario,
        analysis_workers=args.analysis_workers,
        downstream_index_cache_dir=args.downstream_index_cache_dir,
        upstream_file_index_cache_dir=args.upstream_file_index_cache_dir,
        index_workers=args.index_workers,
    )
    outputs = write_reports(report, args.output_dir)
    console_summary = report["summary"]
    if args.scenario == "vllm-interface":
        pr_payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        console_summary = pr_payload["summary"]
    console = {
        "metadata": report["metadata"],
        "summary": console_summary,
        "outputs": outputs,
    }
    print(json.dumps(console, ensure_ascii=False, indent=2))
    actionable_introduced = report["summary"]["actionable_introduced_break"]
    if args.fail_on == "introduced" and actionable_introduced:
        return 1
    if args.fail_on == "unresolved" and (actionable_introduced or report["summary"]["analysis_unresolved"]):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Run the legacy-compatible current-pair generator.")
    _range_parser(subparsers)
    _validate_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "generate":
        generator.main(values[1:])
        return 0
    parser = build_parser()
    args = parser.parse_args(values)
    try:
        if args.command == "analyze-range":
            return _analyze(args)
        if args.command == "validate":
            return _validate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 2
