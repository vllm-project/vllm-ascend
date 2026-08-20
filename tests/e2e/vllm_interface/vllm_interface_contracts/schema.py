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

"""Serialization and comparison helpers for interface-boundary analysis.

This module intentionally knows nothing about source parsing, MRO resolution,
or patch-flow analysis. The public generator module keeps its legacy facade and
passes model objects plus the signature-contract encoder into these helpers.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def relation_payloads(
    relations: Iterable[Any],
    *,
    vllm_sha: str,
    ascend_sha: str,
    findings: Iterable[Any] = (),
    external_sources: dict[str, str] | None = None,
    schema_version: int,
    generator_version: str,
    supported_relations: Iterable[str],
    signature_contract_payload: Callable[[Any], dict[str, object] | None],
) -> list[dict[str, Any]]:
    """Encode normalized relations and findings in the compact JSONL schema."""
    grouped: dict[
        tuple[str, str, str | None, str, str, str | None, str],
        list[Any],
    ] = defaultdict(list)
    for relation in relations:
        signature_key = json.dumps(
            relation.upstream_signature,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        signature_contract_key = json.dumps(
            signature_contract_payload(relation.upstream_signature_contract),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        grouped[
            (
                relation.upstream_package,
                relation.upstream_file,
                relation.upstream_owner,
                relation.upstream_name,
                signature_key,
                relation.upstream_descriptor_kind,
                signature_contract_key,
            )
        ].append(relation)

    payloads: list[dict[str, Any]] = []
    relation_count = 0
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0],
            item[1],
            item[2] or "",
            item[3],
            item[4],
            item[6],
        ),
    ):
        (
            source_package,
            upstream_file,
            owner,
            name,
            signature_key,
            upstream_descriptor_kind,
            signature_contract_key,
        ) = key
        consumers = []
        evidence_records = []
        for relation in sorted(
            grouped[key],
            key=lambda item: (
                item.relation,
                item.downstream_file,
                item.downstream_owner or "",
                item.downstream_name,
            ),
        ):
            consumers.append(
                [
                    relation.relation,
                    relation.downstream_file,
                    relation.downstream_owner,
                    relation.downstream_name,
                    relation.downstream_signature,
                    relation.downstream_descriptor_kind,
                    relation.installed_descriptor_kind,
                    signature_contract_payload(relation.downstream_signature_contract),
                    signature_contract_payload(relation.installed_signature_contract),
                ]
            )
            evidence_records.append(
                {
                    "consumer": [
                        relation.relation,
                        relation.downstream_file,
                        relation.downstream_owner,
                        relation.downstream_name,
                    ],
                    "occurrences": [evidence.as_dict() for evidence in relation.evidence],
                }
            )
            relation_count += 1
        payloads.append(
            {
                "p": source_package,
                "u": [
                    upstream_file,
                    owner,
                    name,
                    json.loads(signature_key),
                    upstream_descriptor_kind,
                    json.loads(signature_contract_key),
                ],
                "c": consumers,
                "e": evidence_records,
            }
        )

    finding_payloads = [
        {"f": finding.as_dict()}
        for finding in sorted(
            findings,
            key=lambda item: (
                item.status,
                item.reason_code,
                item.relation,
                item.downstream_file,
                item.evidence_line,
                item.target_expression,
            ),
        )
    ]
    finding_statuses = Counter(payload["f"]["status"] for payload in finding_payloads)
    meta = {
        "_meta": {
            "schema": schema_version,
            "generator": generator_version,
            "vllm": vllm_sha,
            "vllm_ascend": ascend_sha,
            "external_sources": dict(sorted((external_sources or {}).items())),
            "contracts": len(payloads),
            "relations": relation_count,
            "findings": len(finding_payloads),
            "findings_by_status": dict(sorted(finding_statuses.items())),
            "scope": sorted(supported_relations),
        }
    }
    return [meta, *payloads, *finding_payloads]


def write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    """Write compact, deterministic JSONL output."""
    text = "\n".join(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def relation_label(relation: Any) -> dict[str, Any]:
    return {
        "relation": relation.relation,
        "upstream": {
            "package": relation.upstream_package,
            "file": relation.upstream_file,
            "owner": relation.upstream_owner,
            "name": relation.upstream_name,
        },
        "downstream": {
            "file": relation.downstream_file,
            "owner": relation.downstream_owner,
            "name": relation.downstream_name,
        },
    }


def downstream_label(key: tuple[str, str, str, str]) -> dict[str, Any]:
    return {
        "relation": key[0],
        "file": key[1],
        "owner": key[2] or None,
        "name": key[3],
    }


def _first_exact_match(
    generated: Sequence[Any],
    baseline_relation: Any,
) -> Any | None:
    """Keep the legacy first-match behavior for comparison aliases."""
    baseline_aliases = baseline_relation.comparison_exact_keys()
    return next(
        (
            relation
            for relation in generated
            if any(alias in relation.comparison_exact_keys() for alias in baseline_aliases)
        ),
        None,
    )


def compare_relations(
    generated: Sequence[Any],
    baseline: Sequence[Any],
    findings: Sequence[Any],
    *,
    signature_contract_payload: Callable[[Any], dict[str, object] | None],
) -> dict[str, Any]:
    """Compare generated relations with one compact-table baseline."""
    finding_statuses = Counter(finding.status for finding in findings)
    generated_exact = {relation.exact_key(): relation for relation in generated}
    baseline_exact = {relation.exact_key(): relation for relation in baseline}
    generated_exact_aliases = {key: relation for relation in generated for key in relation.comparison_exact_keys()}
    baseline_exact_aliases = {key: relation for relation in baseline for key in relation.comparison_exact_keys()}
    generated_downstream: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    baseline_downstream: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for relation in generated:
        for key in relation.comparison_downstream_keys():
            generated_downstream[key].append(relation)
    for relation in baseline:
        for key in relation.comparison_downstream_keys():
            baseline_downstream[key].append(relation)

    exact_matches = {
        key
        for key in baseline_exact
        if any(alias in generated_exact_aliases for alias in baseline_exact[key].comparison_exact_keys())
    }
    matched_relations = {key: _first_exact_match(generated, baseline_exact[key]) for key in exact_matches}

    descriptor_kind_changes = []
    for key in sorted(exact_matches):
        baseline_relation = baseline_exact[key]
        baseline_kinds = (
            baseline_relation.upstream_descriptor_kind,
            baseline_relation.downstream_descriptor_kind,
            baseline_relation.installed_descriptor_kind,
        )
        if baseline_kinds == (None, None, None):
            continue
        generated_relation = matched_relations[key]
        if generated_relation is None:
            continue
        generated_kinds = (
            generated_relation.upstream_descriptor_kind,
            generated_relation.downstream_descriptor_kind,
            generated_relation.installed_descriptor_kind,
        )
        if generated_kinds != baseline_kinds:
            descriptor_kind_changes.append(
                {
                    "relation": relation_label(generated_relation),
                    "baseline": list(baseline_kinds),
                    "generated": list(generated_kinds),
                }
            )

    signature_contract_changes = []
    for key in sorted(exact_matches):
        baseline_relation = baseline_exact[key]
        baseline_contracts = (
            baseline_relation.upstream_signature_contract,
            baseline_relation.downstream_signature_contract,
            baseline_relation.installed_signature_contract,
        )
        if baseline_contracts == (None, None, None):
            continue
        generated_relation = matched_relations[key]
        if generated_relation is None:
            continue
        generated_contracts = (
            generated_relation.upstream_signature_contract,
            generated_relation.downstream_signature_contract,
            generated_relation.installed_signature_contract,
        )
        if generated_contracts == baseline_contracts:
            continue
        signature_contract_changes.append(
            {
                "relation": relation_label(generated_relation),
                "baseline": {
                    "upstream": signature_contract_payload(baseline_contracts[0]),
                    "downstream": signature_contract_payload(baseline_contracts[1]),
                    "installed": signature_contract_payload(baseline_contracts[2]),
                },
                "generated": {
                    "upstream": signature_contract_payload(generated_contracts[0]),
                    "downstream": signature_contract_payload(generated_contracts[1]),
                    "installed": signature_contract_payload(generated_contracts[2]),
                },
            }
        )

    different_upstream = []
    baseline_downstream_keys = {relation.downstream_key() for relation in baseline}
    for key in sorted(baseline_downstream_keys & set(generated_downstream)):
        generated_targets = sorted(relation.upstream_key() for relation in generated_downstream[key])
        baseline_targets = sorted(relation.upstream_key() for relation in baseline_downstream[key])
        if generated_targets != baseline_targets:
            different_upstream.append(
                {
                    "downstream": {
                        "relation": key[0],
                        "file": key[1],
                        "owner": key[2] or None,
                        "name": key[3],
                    },
                    "baseline_upstream": baseline_targets,
                    "generated_upstream": generated_targets,
                }
            )

    old_only_keys = set(baseline_exact) - exact_matches
    new_only_keys = {
        key
        for key, relation in generated_exact.items()
        if not any(alias in baseline_exact_aliases for alias in relation.comparison_exact_keys())
    }
    generated_downstream_keys = {relation.downstream_key() for relation in generated}
    covered_downstream_keys = {key for key in baseline_downstream_keys if key in generated_downstream}
    missing_downstream_keys = baseline_downstream_keys - covered_downstream_keys
    new_downstream_keys = {
        key
        for key in generated_downstream_keys
        if not any(
            alias in baseline_downstream
            for relation in generated
            if relation.downstream_key() == key
            for alias in relation.comparison_downstream_keys()
        )
    }
    downstream_coverage = (
        len(covered_downstream_keys) / len(baseline_downstream_keys) * 100 if baseline_downstream_keys else 100.0
    )
    return {
        "summary": {
            "generated_relations": len(generated),
            "baseline_relations": len(baseline),
            "exact_matches": len(exact_matches),
            "descriptor_kind_changes": len(descriptor_kind_changes),
            "signature_contract_changes": len(signature_contract_changes),
            "same_downstream_different_upstream": len(different_upstream),
            "old_only": len(old_only_keys),
            "new_only": len(new_only_keys),
            "findings": len(findings),
            "unresolved": finding_statuses["review"],
            "upstream_risks": finding_statuses["risk"],
            "expected": finding_statuses["expected"],
            "excluded": finding_statuses["excluded"],
            "verified_findings": finding_statuses["verified"],
            "generator_issues": sum(finding.generator_issue for finding in findings),
            "generated_downstream_endpoints": len(generated_downstream_keys),
            "baseline_downstream_endpoints": len(baseline_downstream_keys),
            "covered_downstream_endpoints": len(covered_downstream_keys),
            "missing_downstream_endpoints": len(missing_downstream_keys),
            "new_downstream_endpoints": len(new_downstream_keys),
            "downstream_coverage_percent": round(downstream_coverage, 2),
            "generated_by_relation": dict(sorted(Counter(relation.relation for relation in generated).items())),
            "baseline_by_relation": dict(sorted(Counter(relation.relation for relation in baseline).items())),
        },
        "same_downstream_different_upstream": different_upstream,
        "descriptor_kind_changes": descriptor_kind_changes,
        "signature_contract_changes": signature_contract_changes,
        "old_only": [relation_label(baseline_exact[key]) for key in sorted(old_only_keys)],
        "new_only": [relation_label(generated_exact[key]) for key in sorted(new_only_keys)],
        "missing_downstream": [downstream_label(key) for key in sorted(missing_downstream_keys)],
        "new_downstream": [downstream_label(key) for key in sorted(new_downstream_keys)],
        "findings": [finding.as_dict() for finding in findings],
    }
