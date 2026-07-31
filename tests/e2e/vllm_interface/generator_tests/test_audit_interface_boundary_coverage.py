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

import hashlib
import importlib.util
import json
import subprocess
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


def _commit_fixture_repo(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Interface Auditor",
            "-c",
            "user.email=interface-auditor@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_snapshot_manifest(root: Path, package: str, commit: str) -> None:
    package_root = root.joinpath(*package.split("."))
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }
    _write(
        root,
        ".interface-source.json",
        json.dumps(
            {
                "schema": 1,
                "package": package,
                "commit": commit,
                "files": files,
            }
        ),
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


def test_expected_shas_verify_mapping_and_exact_git_source_roots(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-源码"
    ascend_root = tmp_path / "ascend-源码"
    external_root = tmp_path / "external-源码"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(external_root, "external/__init__.py", "")
    vllm_sha = _commit_fixture_repo(vllm_root)
    ascend_sha = _commit_fixture_repo(ascend_root)
    external_sha = _commit_fixture_repo(external_root)
    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(
        mapping,
        [
            {
                "_meta": {
                    "vllm": vllm_sha,
                    "vllm_ascend": ascend_sha,
                    "external_sources": {"external": external_sha},
                }
            }
        ],
    )

    report = auditor.audit_mapping_coverage(
        vllm_root,
        ascend_root,
        mapping,
        external_roots={"external": external_root},
        expect_vllm_sha=vllm_sha,
        expect_ascend_sha=ascend_sha,
        expect_external_shas={"external": external_sha},
    )

    assert report["_meta"]["verified_sources"] == {
        "vllm": vllm_sha,
        "vllm_ascend": ascend_sha,
        "external_sources": {"external": external_sha},
    }

    claimed_sha = "claimed-by-mapping"
    _write_jsonl(
        mapping,
        [
            {
                "_meta": {
                    "vllm": claimed_sha,
                    "vllm_ascend": ascend_sha,
                    "external_sources": {"external": external_sha},
                }
            }
        ],
    )
    with pytest.raises(ValueError, match="vLLM source SHA mismatch"):
        auditor.audit_mapping_coverage(
            vllm_root,
            ascend_root,
            mapping,
            external_roots={"external": external_root},
            expect_vllm_sha=claimed_sha,
            expect_ascend_sha=ascend_sha,
            expect_external_shas={"external": external_sha},
        )


def test_external_snapshot_verifies_mapping_file_list_and_digests(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    external_root = tmp_path / "external-snapshot"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    source = external_root / "external" / "base.py"
    _write(external_root, "external/base.py", "class ExternalBase:\n    pass\n")
    external_sha = "external-source-sha"
    _write_snapshot_manifest(external_root, "external", external_sha)
    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(
        mapping,
        [{"_meta": {"external_sources": {"external": external_sha}}}],
    )

    report = auditor.audit_mapping_coverage(
        vllm_root,
        ascend_root,
        mapping,
        external_roots={"external": external_root},
        expect_external_shas={"external": external_sha},
    )
    assert report["_meta"]["verified_sources"]["external_sources"] == {"external": external_sha}

    _write_jsonl(
        mapping,
        [{"_meta": {"external_sources": {"external": "wrong-source"}}}],
    )
    with pytest.raises(ValueError, match="mapping external source SHA mismatch"):
        auditor.audit_mapping_coverage(
            vllm_root,
            ascend_root,
            mapping,
            external_roots={"external": external_root},
            expect_external_shas={"external": external_sha},
        )
    _write_jsonl(
        mapping,
        [{"_meta": {"external_sources": {"external": external_sha}}}],
    )

    source.write_text("class Changed:\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        auditor.audit_mapping_coverage(
            vllm_root,
            ascend_root,
            mapping,
            external_roots={"external": external_root},
            expect_external_shas={"external": external_sha},
        )

    source.write_text("class ExternalBase:\n    pass\n", encoding="utf-8")
    _write(external_root, "external/extra.py", "")
    with pytest.raises(ValueError, match="file set changed"):
        auditor.audit_mapping_coverage(
            vllm_root,
            ascend_root,
            mapping,
            external_roots={"external": external_root},
            expect_external_shas={"external": external_sha},
        )


def test_external_root_enables_strict_external_override_resolution(
    tmp_path: Path,
) -> None:
    torch_root = tmp_path / "pytorch"
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(torch_root, "torch/__init__.py", "")
    _write(torch_root, "torch/nn/__init__.py", "from .modules import *\n")
    _write(
        torch_root,
        "torch/nn/modules/__init__.py",
        "from .module import Module\n",
    )
    _write(
        torch_root,
        "torch/nn/modules/module.py",
        """
class Module:
    def forward(self, value):
        return value
""",
    )
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/model.py",
        """
import torch.nn as nn


class Interface:
    pass


class Model(nn.Module, Interface):
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.model import Model


class Child(Model):
    def forward(self, value):
        return value
""",
    )

    without_external = auditor.IndependentCandidateScanner(
        vllm_root,
        ascend_root,
    )
    assert not [candidate for candidate in without_external.scan() if candidate.relation == "override"]
    assert not without_external._strict_mro("vllm_ascend.plugin.Child").complete

    with_external = auditor.IndependentCandidateScanner(
        vllm_root,
        ascend_root,
        external_roots={"torch": torch_root},
    )
    overrides = [candidate for candidate in with_external.scan() if candidate.relation == "override"]

    assert with_external._strict_mro("vllm_ascend.plugin.Child").complete
    assert len(overrides) == 1
    assert overrides[0].targets == ("torch.nn.modules.module.Module.forward",)
    assert overrides[0].kinds == ("external_override",)


def test_known_stdlib_structural_bases_complete_c3_without_guessing(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/model.py",
        """
from abc import ABC
from typing import Generic, Protocol, TypeVar


T = TypeVar("T")


class Base(ABC, Generic[T]):
    def hook(self):
        pass


class Contract(Protocol[T]):
    def protocol_hook(self):
        pass


class Combined(Base[T], Contract[T]):
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.model import Combined


class Child(Combined):
    def hook(self):
        pass

    def protocol_hook(self):
        pass
""",
    )

    scanner = auditor.IndependentCandidateScanner(vllm_root, ascend_root)
    overrides = [candidate for candidate in scanner.scan() if candidate.relation == "override"]
    mro = scanner._strict_mro("vllm_ascend.plugin.Child")

    assert mro.complete
    assert mro.owners == (
        "vllm_ascend.plugin.Child",
        "vllm.model.Combined",
        "vllm.model.Base",
        "abc.ABC",
        "vllm.model.Contract",
        "typing.Protocol",
        "typing.Generic",
    )
    assert {candidate.targets for candidate in overrides} == {
        ("vllm.model.Base.hook",),
        ("vllm.model.Contract.protocol_hook",),
    }


def test_strict_c3_does_not_skip_a_downstream_effective_owner(
    tmp_path: Path,
) -> None:
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
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Mid(Base):
    def run(self, value):
        return value


class Child(Mid):
    def run(self, value):
        return value
""",
    )

    scanner = auditor.IndependentCandidateScanner(vllm_root, ascend_root)
    overrides = [candidate for candidate in scanner.scan() if candidate.relation == "override"]

    assert scanner._strict_mro("vllm_ascend.plugin.Child").owners == (
        "vllm_ascend.plugin.Child",
        "vllm_ascend.plugin.Mid",
        "vllm.base.Base",
    )
    assert len(overrides) == 1
    assert overrides[0].line == 6
    assert overrides[0].targets == ("vllm.base.Base.run",)


def test_incomplete_mro_is_reported_without_selecting_an_owner(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        "class Base:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from unknown_package import Mixin
from vllm.base import Base


class Child(Mixin, Base):
    def run(self):
        pass
""",
    )

    scanner = auditor.IndependentCandidateScanner(vllm_root, ascend_root)
    overrides = [candidate for candidate in scanner.scan() if candidate.relation == "override"]
    mro = scanner._strict_mro("vllm_ascend.plugin.Child")

    assert not mro.complete
    assert "not indexed" in mro.reason
    assert len(overrides) == 1
    assert overrides[0].kinds == ("incomplete_mro",)
    assert overrides[0].targets == ("vllm.base.Base.run",)


def test_missing_upstream_super_target_is_an_override_candidate_not_an_orphan(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Base:\n    pass\n")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return super().run(value)
""",
    )

    candidates = auditor.IndependentCandidateScanner(vllm_root, ascend_root).scan()
    override = next(candidate for candidate in candidates if candidate.relation == "override")

    assert override.line == 6
    assert override.targets == ("vllm.base.Base.run",)
    assert override.kinds == ("missing_upstream_super_target",)

    mapping = tmp_path / "mapping.jsonl"
    _write_jsonl(
        mapping,
        [
            {"_meta": {}},
            {
                "f": {
                    "relation": "override",
                    "downstream": {
                        "file": override.file,
                        "owner": "Child",
                        "name": "run",
                    },
                    "target_expression": "vllm.base.Base.run",
                    "evidence": {"file": override.file, "line": override.line},
                    "status": "risk",
                    "reason_code": "missing_upstream_super_target",
                    "generator_issue": False,
                    "reason": "fixture",
                }
            },
            *[_relation_payload(candidate) for candidate in candidates if candidate.site_key != override.site_key],
        ],
    )

    report = auditor.audit_mapping_coverage(vllm_root, ascend_root, mapping)

    assert report["summary"]["missing"] == 0
    assert report["summary"]["orphan"] == 0


def test_missing_super_candidate_excludes_non_direct_or_unreachable_calls(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Base:\n    pass\n")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def __init__(self):
        super().__init__()

    def dead_branch(self):
        if False:
            return super().dead_branch()

    def after_return(self):
        return None
        return super().after_return()

    def nested_function(self):
        def deferred():
            return super().nested_function()
        return deferred

    def different_name(self):
        return super().other_name()

    def explicit_super(self):
        return super(Base, self).explicit_super()
""",
    )

    overrides = [
        candidate
        for candidate in auditor.IndependentCandidateScanner(vllm_root, ascend_root).scan()
        if candidate.relation == "override"
    ]

    assert not overrides


def test_external_reexport_is_not_a_vllm_owned_patch(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/reexport.py",
        "import third_party_runtime as runtime\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.reexport import runtime


def replacement():
    pass


runtime.hook = replacement
""",
    )

    candidates = auditor.IndependentCandidateScanner(
        vllm_root,
        ascend_root,
    ).scan()

    assert not [candidate for candidate in candidates if candidate.relation == "monkey_patch"]


def test_external_root_cli_values_are_validated(tmp_path: Path) -> None:
    roots = auditor._parse_external_roots([f"torch={tmp_path / 'torch'}", f"external={tmp_path / 'external'}"])
    assert set(roots) == {"torch", "external"}
    assert auditor._parse_external_shas(["torch=source-sha"]) == {"torch": "source-sha"}
    with pytest.raises(ValueError, match="expected PACKAGE=VALUE"):
        auditor._parse_external_roots(["torch"])
    with pytest.raises(ValueError, match="duplicate"):
        auditor._parse_external_roots(["torch=first", "torch=second"])
    with pytest.raises(ValueError, match="same packages"):
        auditor._verify_source_inputs(
            tmp_path,
            tmp_path,
            {"torch": tmp_path / "torch"},
            expect_vllm_sha=None,
            expect_ascend_sha=None,
            expect_external_shas={},
        )
