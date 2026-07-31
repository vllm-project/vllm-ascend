# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_AUDITOR_PATH = Path(__file__).parents[1] / "audit_interface_boundary_coverage.py"
_SPEC = importlib.util.spec_from_file_location("interface_boundary_coverage_auditor", _AUDITOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
auditor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = auditor
_SPEC.loader.exec_module(auditor)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(payload, separators=(",", ":")) for payload in payloads) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def source_pair(tmp_path: Path) -> tuple[Path, Path]:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    def run(self, value):
        return value


class PatchTarget:
    def hook(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base
from vllm.base import PatchTarget
from vllm.missing import MissingBase


def replacement(self, value):
    return value


def alias_run(self, value):
    return value


PatchTarget.hook = replacement
setattr(PatchTarget, "injected", replacement)


def install_local_patch():
    from vllm.base import PatchTarget as LocalTarget

    LocalTarget.hook = replacement


class Child(Base):
    def run(self, value):
        return value


class AliasChild(Base):
    run = alias_run


class BrokenChild(MissingBase):
    pass
""",
    )
    return vllm_root, ascend_root


def _relation_payload(candidate) -> dict:
    occurrence = {
        "file": candidate.file,
        "line": candidate.line,
    }
    if candidate.scope:
        occurrence["scope"] = candidate.scope
    return {
        "u": ["vllm/base.py", "Base", "symbol", None],
        "c": [[candidate.relation, candidate.file, None, "consumer", None]],
        "e": [
            {
                "consumer": [candidate.relation, candidate.file, None, "consumer"],
                "occurrences": [occurrence],
            }
        ],
    }


def test_independent_scanner_enumerates_supported_candidate_shapes(
    source_pair: tuple[Path, Path],
) -> None:
    vllm_root, ascend_root = source_pair
    candidates = auditor.IndependentCandidateScanner(vllm_root, ascend_root).scan()

    assert sum(candidate.relation == "monkey_patch" for candidate in candidates) == 3
    assert sum(candidate.relation == "inheritance" for candidate in candidates) == 3
    assert sum(candidate.relation == "override" for candidate in candidates) == 2
    assert any(
        candidate.relation == "monkey_patch" and candidate.scope == "install_local_patch" for candidate in candidates
    )
    assert any(candidate.relation == "override" and "callable_alias" in candidate.kinds for candidate in candidates)
    assert any(
        candidate.relation == "inheritance"
        and any("vllm.missing.MissingBase" in target for target in candidate.targets)
        for candidate in candidates
    )
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_clean_mapping_classifies_every_candidate_once(
    source_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = source_pair
    candidates = auditor.IndependentCandidateScanner(vllm_root, ascend_root).scan()
    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(
        mapping,
        [
            {"_meta": {"vllm": "upstream", "vllm_ascend": "downstream"}},
            *[_relation_payload(candidate) for candidate in candidates],
        ],
    )

    report = auditor.audit_mapping_coverage(
        vllm_root,
        ascend_root,
        mapping,
        expect_vllm_sha="upstream",
        expect_ascend_sha="downstream",
    )

    assert report["summary"] == {
        "candidates": len(candidates),
        "classified": len(candidates),
        "missing": 0,
        "conflicting": 0,
        "orphan": 0,
        "generator_issue_review": 0,
    }


def test_audit_reports_missing_conflicting_orphan_and_generator_issue(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/core.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.core import Target


def replacement(self):
    pass


Target.run = replacement
""",
    )
    candidate = auditor.IndependentCandidateScanner(vllm_root, ascend_root).scan()[0]
    occurrence = {"file": candidate.file, "line": candidate.line}
    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(
        mapping,
        [
            {"_meta": {"vllm": "upstream", "vllm_ascend": "downstream"}},
            _relation_payload(candidate),
            {
                "f": {
                    "relation": "monkey_patch",
                    "downstream": {"file": candidate.file, "owner": None, "name": "replacement"},
                    "target_expression": "vllm.core.Target.run",
                    "evidence": occurrence,
                    "status": "review",
                    "reason_code": "dynamic_target",
                    "generator_issue": True,
                    "reason": "fixture",
                }
            },
            {
                "f": {
                    "relation": "monkey_patch",
                    "downstream": {"file": candidate.file, "owner": None, "name": "ghost"},
                    "target_expression": "vllm.core.Target.ghost",
                    "evidence": {"file": candidate.file, "line": 999},
                    "status": "risk",
                    "reason_code": "missing_upstream_callable",
                    "generator_issue": False,
                    "reason": "fixture",
                }
            },
        ],
    )

    report = auditor.audit_mapping_coverage(vllm_root, ascend_root, mapping)

    assert report["summary"]["missing"] == 0
    assert report["summary"]["conflicting"] == 1
    assert report["summary"]["orphan"] == 1
    assert report["summary"]["generator_issue_review"] == 1

    empty_mapping = tmp_path / "empty.jsonl"
    _write_jsonl(empty_mapping, [{"_meta": {}}])
    missing_report = auditor.audit_mapping_coverage(vllm_root, ascend_root, empty_mapping)
    assert missing_report["summary"]["missing"] == 1


def test_sha_mismatch_fails_before_reporting(
    source_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = source_pair
    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(mapping, [{"_meta": {"vllm": "actual"}}])

    with pytest.raises(ValueError, match="mapping vLLM SHA mismatch"):
        auditor.audit_mapping_coverage(
            vllm_root,
            ascend_root,
            mapping,
            expect_vllm_sha="expected",
        )
