# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score independently reviewed contracts and summarize unresolved evidence.

This module consumes JSON reports; it does not discover source dependencies.
Unlabelled findings are unassessed, never automatically false positives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA_VERSION = 1
IDENTITY_FIELDS = ("vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha", "external_sources", "scenario", "profile")
EVIDENCE_KINDS = ("runtime_contract", "interface_alignment", "historical", "negative_control")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _symbol(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("details", {}).get("root_upstream") or item["upstream"]["old"]


def _matches(item: dict[str, Any], selector: dict[str, Any]) -> bool:
    expanded = {**item, "upstream_symbol": _symbol(item)}
    return all(_get(expanded, key) == value for key, value in selector.items())


def _actionable(item: dict[str, Any]) -> bool:
    return item.get("classification") == "introduced_break" and item.get("action") == "modify"


def validate_report(report: dict[str, Any]) -> None:
    metadata = report.get("metadata", {})
    if metadata.get("scenario") != "main2main":
        raise ValueError("Benchmark requires a full main2main report, not an upstream-CI report")
    if not metadata.get("range_analyzer_version"):
        raise ValueError("Report must identify its analyzer version")
    capabilities = metadata.get("analysis_plan", {}).get("capabilities", {})
    for name in ("monkey_patch", "direct_attribute", "override", "direct_import", "direct_call", "inherited_state"):
        if capabilities.get(name, {}).get("state") != "analyzed":
            raise ValueError(f"Required full-analysis capability was not analyzed: {name}")
    findings = report.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Report findings must be a list")
    ids = [item.get("id") for item in findings]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Report finding IDs must be present and unique")
    if any(not item.get("root_cause_id") for item in findings):
        raise ValueError("Report findings must include root-cause IDs")


def validate_manifest(manifest: dict[str, Any], report: dict[str, Any]) -> None:
    validate_report(report)
    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark manifest schema")
    inputs = manifest.get("inputs", {})
    for key in IDENTITY_FIELDS:
        if key not in inputs or key not in report["metadata"] or inputs[key] != report["metadata"][key]:
            raise ValueError(f"Benchmark/report input mismatch: {key}")
    for key in ("vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha"):
        value = inputs[key]
        if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Expected a full commit SHA: {key}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must contain independently reviewed cases")
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Benchmark case IDs must be present and unique")
        ids.add(case_id)
        if not case.get("rationale") or not case.get("evidence") or case.get("evidence_kind") not in EVIDENCE_KINDS:
            raise ValueError(f"Case requires rationale, independent evidence and evidence kind: {case_id}")
        if not isinstance(case.get("root_count"), int) or case["root_count"] < 0:
            raise ValueError(f"Case requires a nonnegative root_count: {case_id}")
        if not case.get("checks"):
            raise ValueError(f"Case requires checks: {case_id}")
        for check in case["checks"]:
            selector = check.get("match", {})
            if not selector or not isinstance(check.get("count"), int) or check["count"] < 0:
                raise ValueError(f"Check requires a selector and nonnegative count: {case_id}")
            # Finding IDs and decisions would make a changed decision disappear
            # from the sample, or tie the oracle to one analyzer's output.
            forbidden = {"id", "root_cause_id", "classification", "priority", "action", "confidence"}
            if forbidden.intersection(selector):
                raise ValueError(f"Selectors must identify source contracts, not report decisions: {case_id}")
            if check["count"] and not all(
                key in check.get("expected", {}) for key in ("classification", "action", "priority")
            ):
                raise ValueError(f"Positive check requires classification, action and priority: {case_id}")


def score_report(manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest, report)
    outcomes: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}
    claimed_roots: dict[str, str] = {}
    true_positive_roots: set[str] = set()
    false_positive_roots: set[str] = set()
    positive_cases = 0
    detected_cases = 0
    for case in manifest["cases"]:
        checks = []
        roots: set[str] = set()
        expected_positive = False
        detected = False
        for check in case["checks"]:
            matches = [item for item in report["findings"] if _matches(item, check["match"])]
            expected = check.get("expected", {})
            positive = check["count"] > 0 and _actionable(expected)
            expected_positive |= positive
            errors = []
            if len(matches) != check["count"]:
                errors.append(f"Expected {check['count']} findings; observed {len(matches)}")
            for item in matches:
                finding_id, root = item["id"], item["root_cause_id"]
                if finding_id in claimed:
                    raise ValueError(f"Overlapping benchmark selectors: {claimed[finding_id]} and {case['id']}")
                claimed[finding_id] = case["id"]
                if root in claimed_roots and claimed_roots[root] != case["id"]:
                    errors.append(f"Root incorrectly shared with independent case {claimed_roots[root]}")
                claimed_roots[root] = case["id"]
                roots.add(root)
                for key, value in expected.items():
                    if _get(item, key) != value:
                        errors.append(f"{finding_id}: {key} expected {value!r}, observed {_get(item, key)!r}")
                if _actionable(item):
                    if positive:
                        true_positive_roots.add(root)
                        detected = True
                    else:
                        false_positive_roots.add(root)
            checks.append(
                {"selector": check["match"], "finding_ids": [item["id"] for item in matches], "errors": errors}
            )
        positive_cases += expected_positive
        detected_cases += expected_positive and detected
        root_ok = len(roots) == case["root_count"]
        outcomes.append(
            {
                "id": case["id"],
                "evidence_kind": case["evidence_kind"],
                "passed": root_ok and not any(check["errors"] for check in checks),
                "expected_root_count": case["root_count"],
                "observed_root_count": len(roots),
                "root_cause_ids": sorted(roots),
                "checks": checks,
            }
        )
    unassessed = [item for item in report["findings"] if _actionable(item) and item["id"] not in claimed]
    reviewed_positive = true_positive_roots | false_positive_roots
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": manifest.get("id"),
        "split": manifest.get("split"),
        "passed": all(item["passed"] for item in outcomes),
        "provenance": {
            "manifest_sha256": _digest(manifest),
            "report_sha256": _digest(report),
            "metadata": report["metadata"],
        },
        "metrics": {
            "cases": len(outcomes),
            "cases_passed": sum(item["passed"] for item in outcomes),
            "expected_actionable_cases": positive_cases,
            "detected_actionable_cases": detected_cases,
            "known_case_recall": detected_cases / positive_cases if positive_cases else None,
            "reviewed_actionable_root_precision": len(true_positive_roots - false_positive_roots)
            / len(reviewed_positive)
            if reviewed_positive
            else None,
            "unassessed_actionable_roots": len({item["root_cause_id"] for item in unassessed}),
            "unassessed_actionable_findings": len(unassessed),
            "scope": (
                "Only independently labelled contracts; unassessed findings are not false positives. "
                "No global accuracy claim."
            ),
        },
        "cases": outcomes,
        "unassessed_actionable": [
            {key: item[key] for key in ("id", "root_cause_id", "relation", "downstream")} for item in unassessed
        ],
    }


def _unresolved_category(item: dict[str, Any]) -> str:
    reason = " ".join(str(item.get("compatibility", {}).get(side, {}).get("reason", "")) for side in ("old", "new"))
    details = item.get("details", {})
    if item["relation"] == "direct_attribute":
        receiver = details.get("receiver_type") or ""
        if receiver.startswith("vllm_ascend.") and not details.get("lookup_root"):
            return "downstream_receiver_without_upstream_lookup_root"
        if details.get("access_kind") == "direct":
            return "module_or_class_attribute_binding"
        return "upstream_instance_field_binding"
    if "MRO" in reason or "external base" in reason:
        return "incomplete_mro_or_external_base"
    if "transform" in reason or "signature variants" in reason:
        return "runtime_signature_transform"
    if "callable kind" in reason:
        return "callable_descriptor_mismatch"
    if "return" in reason:
        return "return_protocol"
    return f"{item['relation']}_binding_or_contract"


def triage_report(report: dict[str, Any]) -> dict[str, Any]:
    validate_report(report)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: Counter[str] = Counter()
    unresolved = [item for item in report["findings"] if item["classification"] == "analysis_unresolved"]
    for item in unresolved:
        details = item.get("details", {})
        category = _unresolved_category(item)
        categories[category] += 1
        key = {
            "category": category,
            "relation": item["relation"],
            "contract_kind": item["contract_kind"],
            "target": details.get("target") or _symbol(item),
            "receiver_type": details.get("receiver_type"),
            "lookup_root": details.get("lookup_root"),
            "compatibility": item.get("compatibility"),
        }
        groups[json.dumps(key, sort_keys=True)].append(item)
    rows = []
    for serialized_key, items in groups.items():
        rows.append(
            {
                **json.loads(serialized_key),
                "findings": len(items),
                "finding_ids": [item["id"] for item in items],
                "root_cause_ids": sorted({item["root_cause_id"] for item in items}),
                "locations": [item["downstream"] for item in items],
            }
        )
    rows.sort(key=lambda item: (-item["findings"], json.dumps(item["target"], sort_keys=True)))
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "provenance": {"report_sha256": _digest(report), "metadata": report["metadata"]},
        "summary": {
            "unresolved_findings": len(unresolved),
            "review_groups": len(rows),
            "by_category": dict(categories.most_common()),
        },
        "interpretation": (
            "Categories explain missing evidence; they do not establish safety or a break. "
            "All original findings are retained."
        ),
        "groups": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "triage"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--report", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
        if command == "score":
            subparser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = _read(args.report)
        result = score_report(_read(args.manifest), report) if args.command == "score" else triage_report(report)
        if args.output.resolve() in {args.report.resolve(), getattr(args, "manifest", args.report).resolve()}:
            raise ValueError("Output must not overwrite report or manifest inputs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "passed": result.get("passed"),
                    **result.get("metrics", result.get("summary", {})),
                }
            )
        )
        return 0 if result.get("passed", True) else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
