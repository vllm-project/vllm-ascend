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

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_GENERATOR_PATH = Path(__file__).parents[1] / "generate_interface_boundaries.py"
_SPEC = importlib.util.spec_from_file_location(
    "interface_boundary_generator",
    _GENERATOR_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
generator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = generator
_SPEC.loader.exec_module(generator)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    def __init__(self, config):
        self.config = config

    def run(self, value, *, mode=None):
        return value


class PatchTarget:
    def hook(self, value):
        return value

    def inherited_hook(self, value):
        return value

    def external_hook(self, value):
        return value

    def run(self, value):
        return value


class PatchChild(PatchTarget):
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/implementation.py",
        """
def external_hook(self, value):
    return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base as VllmBase
from vllm.base import PatchChild
from vllm.base import PatchTarget

from vllm_ascend.implementation import external_hook


def patched_hook(self, value):
    return value


def injected_helper(self):
    return 1


def patched_run(self, value):
    return value + self.helper()


PatchTarget.hook = patched_hook
PatchTarget.missing = patched_hook
PatchTarget.helper = injected_helper
PatchTarget.run = patched_run
if not hasattr(PatchTarget, "injected"):
    PatchTarget.injected = patched_hook
if hasattr(PatchTarget, "removed"):
    PatchTarget.removed = patched_hook
PatchChild.inherited_hook = patched_hook
PatchTarget.external_hook = external_hook
PatchTarget.registry["backend"] = patched_hook
dynamic_name = "hook"
setattr(PatchTarget, dynamic_name, patched_hook)
selected_name = None
if hasattr(PatchTarget, "hook"):
    selected_name = "hook"
elif hasattr(PatchTarget, "old_hook"):
    selected_name = "old_hook"
setattr(PatchTarget, selected_name, patched_hook)
unknown_name = choose_patch_name()
setattr(PatchTarget, unknown_name, patched_hook)


class Child(VllmBase):
    def __init__(self, config):
        super().__init__(config)

    def run(self, value, *, mode=None):
        return value

    def local_only(self):
        return None
""",
    )
    return vllm_root, ascend_root


def test_generates_exact_patch_inheritance_and_override(
    source_pair: tuple[Path, Path],
) -> None:
    vllm_root, ascend_root = source_pair
    relations, unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    relation_keys = {
        (
            relation.relation,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_owner,
            relation.downstream_name,
        )
        for relation in relations
    }
    assert (
        "inheritance",
        None,
        "Base",
        "Child",
        "VllmBase",
    ) in relation_keys
    assert (
        "override",
        "Base",
        "__init__",
        "Child",
        "__init__",
    ) in relation_keys
    assert (
        "override",
        "Base",
        "run",
        "Child",
        "run",
    ) in relation_keys
    assert (
        "monkey_patch",
        "PatchTarget",
        "hook",
        None,
        "patched_hook",
    ) in relation_keys
    hook_patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
        and relation.upstream_owner == "PatchTarget"
        and relation.upstream_name == "hook"
    )
    assert len(hook_patch.evidence) == 3
    imported_patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch" and relation.upstream_name == "external_hook"
    )
    assert imported_patch.downstream_file == ("vllm_ascend/implementation.py")
    assert imported_patch.downstream_name == "external_hook"
    assert imported_patch.evidence[0].file == "vllm_ascend/plugin.py"
    assert imported_patch.evidence[0].line == imported_patch.evidence_line
    assert (
        "monkey_patch",
        "PatchTarget",
        "inherited_hook",
        None,
        "patched_hook",
    ) in relation_keys
    assert not any(relation.downstream_name == "local_only" for relation in relations)

    assert len(unresolved) == 5
    missing_target = next(
        relation
        for relation in unresolved
        if relation.relation == "monkey_patch" and relation.target_expression == "vllm.base.PatchTarget.missing"
    )
    assert missing_target.status == "risk"
    assert missing_target.reason_code == "possible_stale_patch"
    assert not missing_target.generator_issue
    injected = next(
        relation for relation in unresolved if relation.target_expression == "vllm.base.PatchTarget.injected"
    )
    assert injected.status == "expected"
    assert injected.reason_code == "inject_missing_member"
    inactive = next(
        relation for relation in unresolved if relation.target_expression == "vllm.base.PatchTarget.removed"
    )
    assert inactive.status == "excluded"
    assert inactive.reason_code == "inactive_guard"
    reachable_injection = next(
        relation for relation in unresolved if relation.target_expression == "vllm.base.PatchTarget.helper"
    )
    assert reachable_injection.status == "expected"
    assert reachable_injection.reason_code == "inject_missing_member"
    assert (
        sum(
            relation.reason == "dynamic setattr attribute name"
            and relation.target_expression == "vllm.base.PatchTarget"
            and relation.status == "review"
            and relation.reason_code == "dynamic_setattr_name"
            and relation.generator_issue
            for relation in unresolved
        )
        == 1
    )
    assert not any(relation.target_expression == "vllm.base.PatchTarget.registry" for relation in unresolved)


def test_init_without_super_is_still_a_verified_override(
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
    def __init__(self, config):
        self.config = config
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def __init__(self, config):
        self.config = config
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    assert any(relation.relation == "override" and relation.downstream_name == "__init__" for relation in relations)


def test_dataclass_generated_init_has_a_field_derived_signature(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/data.py",
        """
from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar


@dataclass
class Base:
    required: int
    optional: int = 1
    _: KW_ONLY
    keyed: int


@dataclass
class Payload(Base):
    local: int = 2
    factory: list = field(default_factory=list)
    ignored: int = field(init=False)
    class_value: ClassVar[int] = 3
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.data import Payload


def replacement(self, *args, **kwargs):
    return None


Payload.__init__ = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patch = next(relation for relation in relations if relation.relation == "monkey_patch")
    assert patch.upstream_name == "__init__"
    assert patch.upstream_signature == [
        "sync",
        [],
        [
            ["self", True],
            ["required", True],
            ["optional", False],
            ["local", False],
            ["factory", False],
        ],
        None,
        [["keyed", True]],
        None,
    ]
    assert not findings


def test_output_is_deterministic(source_pair: tuple[Path, Path]) -> None:
    vllm_root, ascend_root = source_pair
    first, first_unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    second, second_unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    first_payload = generator._relation_payloads(
        first,
        vllm_sha="upstream",
        ascend_sha="downstream",
        findings=first_unresolved,
    )
    second_payload = generator._relation_payloads(
        second,
        vllm_sha="upstream",
        ascend_sha="downstream",
        findings=second_unresolved,
    )
    assert json.dumps(first_payload, sort_keys=True) == json.dumps(
        second_payload,
        sort_keys=True,
    )
    assert [item.as_dict() for item in first_unresolved] == [item.as_dict() for item in second_unresolved]
    assert sum("f" in payload for payload in first_payload) == 5


def test_comparison_tracks_downstream_coverage_separately() -> None:
    common = {
        "relation": "inheritance",
        "upstream_owner": None,
        "upstream_name": "Base",
        "upstream_signature": None,
        "downstream_file": "vllm_ascend/plugin.py",
        "downstream_owner": "Child",
        "downstream_name": "Base",
        "downstream_signature": None,
        "evidence_file": "vllm_ascend/plugin.py",
        "evidence_line": 3,
    }
    baseline = generator.Relation(
        upstream_file="vllm/base/__init__.py",
        **common,
    )
    generated = generator.Relation(
        upstream_file="vllm/base/implementation.py",
        **common,
    )

    report = generator.compare_relations(
        [generated],
        [baseline],
        [],
    )

    assert report["summary"]["exact_matches"] == 0
    assert report["summary"]["same_downstream_different_upstream"] == 1
    assert report["summary"]["covered_downstream_endpoints"] == 1
    assert report["summary"]["missing_downstream_endpoints"] == 0
    assert report["summary"]["downstream_coverage_percent"] == 100.0


def test_comparison_uses_patch_site_as_a_legacy_alias() -> None:
    common = {
        "relation": "monkey_patch",
        "upstream_file": "vllm/core.py",
        "upstream_owner": "Engine",
        "upstream_name": "step",
        "upstream_signature": None,
        "downstream_owner": None,
        "downstream_name": "replacement",
        "downstream_signature": None,
        "evidence_file": "vllm_ascend/plugin.py",
        "evidence_line": 8,
    }
    baseline = generator.Relation(
        downstream_file="vllm_ascend/plugin.py",
        **common,
    )
    generated = generator.Relation(
        downstream_file="vllm_ascend/implementation.py",
        evidence=(
            generator.RelationEvidence(
                file="vllm_ascend/plugin.py",
                line=8,
            ),
        ),
        **common,
    )

    report = generator.compare_relations(
        [generated],
        [baseline],
        [],
    )

    assert report["summary"]["exact_matches"] == 1
    assert report["summary"]["missing_downstream_endpoints"] == 0
    assert report["summary"]["new_downstream_endpoints"] == 0


def test_local_definition_shadows_an_imported_name(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    pass
""",
    )
    _write(
        vllm_root,
        "vllm/wrapper.py",
        """
from vllm.base import Base


class Base(Base):
    pass
""",
    )

    index = generator.RepositoryIndex(vllm_root, "vllm")

    assert index.canonical_name("vllm.wrapper.Base") == ("vllm.wrapper.Base")
    assert index.find_class("vllm.wrapper.Base").file == ("vllm/wrapper.py")
    assert index.find_class("vllm.wrapper.Base").resolved_bases == ("vllm.base.Base",)


def test_star_reexport_resolves_to_the_defining_callable(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "from .core import *\n")
    _write(
        vllm_root,
        "vllm/core.py",
        "def exported(value):\n    return value\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm


def replacement(value):
    return value


vllm.exported = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patch = next(relation for relation in relations if relation.relation == "monkey_patch")
    assert patch.upstream_file == "vllm/core.py"
    assert patch.upstream_name == "exported"
    assert not findings


def test_typed_lazy_export_resolves_to_its_interface_owner(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/platforms/interface.py",
        """
class Platform:
    def verify(self, value):
        return value
""",
    )
    _write(
        vllm_root,
        "vllm/platforms/__init__.py",
        """
from typing import TYPE_CHECKING
from .interface import Platform

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name):
    if name == "current_platform":
        return Platform()
    raise AttributeError(name)
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.platforms import current_platform


def replacement(value):
    return value


current_platform.verify = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patch = next(relation for relation in relations if relation.relation == "monkey_patch")
    assert patch.upstream_file == "vllm/platforms/interface.py"
    assert patch.upstream_owner == "Platform"
    assert patch.upstream_name == "verify"
    assert patch.evidence[0].target_expression == "current_platform.verify"
    assert not findings


def test_main_skips_exact_tag_patch_branches(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class PatchTarget:
    def first(self):
        pass

    def second(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import PatchTarget


def release_patch(self):
    pass


def main_patch(self):
    pass


if vllm_version_is("0.25.1"):
    PatchTarget.first = release_patch
else:
    PatchTarget.first = main_patch

is_release = vllm_version_is("0.25.1")
if is_release:
    PatchTarget.second = release_patch
if not is_release:
    PatchTarget.second = main_patch
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 2
    assert all(relation.downstream_name == "main_patch" for relation in patches)


def test_main_selects_main_import_branch(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/main_base.py",
        "class MainBase:\n    pass\n",
    )
    _write(
        vllm_root,
        "vllm/release_base.py",
        "class ReleaseBase:\n    pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
if vllm_version_is("0.25.1"):
    from vllm.release_base import ReleaseBase as SelectedBase
else:
    from vllm.main_base import MainBase as SelectedBase


class Child(SelectedBase):
    pass
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    inheritance = next(relation for relation in relations if relation.relation == "inheritance")
    assert inheritance.upstream_name == "MainBase"
    assert inheritance.downstream_name == "SelectedBase"


def test_incomplete_owned_mro_is_not_guessed(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
import external


class Base:
    def run(self):
        pass


class Partial(external.Mixin, Base):
    def run(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base
from vllm.base import Partial
from vllm.missing import Missing
import external


class Child(Missing, Base):
    def run(self):
        pass


class OpaqueFirst(external.Mixin, Base):
    def run(self):
        pass


class SafePrefix(Partial):
    def run(self):
        pass
""",
    )

    relations, unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not any(
        relation.relation == "override" and relation.downstream_owner in {"Child", "OpaqueFirst"}
        for relation in relations
    )
    assert any(
        relation.relation == "override" and relation.reason.startswith("incomplete MRO") for relation in unresolved
    )
    assert any(
        relation.relation == "override"
        and relation.downstream_owner == "OpaqueFirst"
        and "opaque or unresolved base" in relation.reason
        for relation in unresolved
    )
    assert any(
        relation.relation == "override"
        and relation.upstream_owner == "Partial"
        and relation.downstream_owner == "SafePrefix"
        and relation.downstream_name == "run"
        for relation in relations
    )


def test_missing_method_on_external_base_is_review_not_upstream_risk(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/model.py",
        """
from external import ExternalBase


class Model(ExternalBase):
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.model import Model


def patched_to(self, *args, **kwargs):
    return self


Model.to = patched_to
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    assert findings[0].status == "review"
    assert findings[0].reason_code == "external_inherited_method"
    assert not findings[0].generator_issue


def test_exact_external_source_completes_patch_and_override_mro(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    external_root = tmp_path / "external-repo"
    _write(external_root, "external/__init__.py", "from .module import ExternalBase\n")
    _write(
        external_root,
        "external/module.py",
        """
class ExternalBase:
    def forward(self, value):
        return value

    def external_only(self, value):
        return value

    def to(self, *args, **kwargs):
        return self
""",
    )
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/model.py",
        """
from external import ExternalBase


class Protocol:
    def forward(self, value):
        return value

    def protocol(self, value):
        return value


class Model(ExternalBase, Protocol):
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.model import Model


def patched_to(self, *args, **kwargs):
    return self


Model.to = patched_to


class Child(Model):
    def forward(self, value):
        return value

    def external_only(self, value):
        return value

    def protocol(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        {"external": external_root},
    ).generate()
    by_endpoint = {
        (
            relation.relation,
            relation.downstream_owner,
            relation.downstream_name,
        ): relation
        for relation in relations
    }

    patch = by_endpoint[("monkey_patch", None, "patched_to")]
    assert patch.upstream_package == "external"
    assert patch.upstream_file == "external/module.py"
    assert patch.upstream_owner == "ExternalBase"
    assert patch.upstream_name == "to"

    assert ("override", "Child", "forward") not in by_endpoint
    assert ("override", "Child", "external_only") not in by_endpoint

    vllm_override = by_endpoint[("override", "Child", "protocol")]
    assert vllm_override.upstream_package == "vllm"
    assert vllm_override.upstream_owner == "Protocol"
    assert any(
        finding.reason_code == "external_override_owner"
        and finding.downstream_owner == "Child"
        and finding.downstream_name == "forward"
        and finding.target_expression == "vllm.model.Protocol.forward"
        for finding in findings
    )
    assert any(
        finding.reason_code == "external_only_override"
        and finding.downstream_owner == "Child"
        and finding.downstream_name == "external_only"
        and finding.target_expression == ("external.module.ExternalBase.external_only")
        for finding in findings
    )

    payloads = generator._relation_payloads(
        relations,
        vllm_sha="vllm-sha",
        ascend_sha="ascend-sha",
        findings=findings,
        external_sources={"external": "external-sha"},
    )
    external_records = [payload for payload in payloads if payload.get("p") == "external"]
    assert len(external_records) == 1
    assert payloads[0]["_meta"]["external_sources"] == {"external": "external-sha"}


def test_unknown_parent_inside_external_source_keeps_mro_in_review(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    external_root = tmp_path / "external-repo"
    _write(external_root, "external/__init__.py", "from .module import ExternalBase\n")
    _write(
        external_root,
        "external/module.py",
        """
import unknown


class ExternalBase(unknown.Parent):
    pass
""",
    )
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/model.py",
        """
from external import ExternalBase


class Protocol:
    def hook(self):
        pass


class Model(ExternalBase, Protocol):
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
    def hook(self):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        {"external": external_root},
    ).generate()

    assert not any(
        relation.relation == "override" and relation.downstream_owner == "Child" and relation.downstream_name == "hook"
        for relation in relations
    )
    review = next(
        finding for finding in findings if finding.downstream_owner == "Child" and finding.downstream_name == "hook"
    )
    assert review.status == "review"
    assert review.reason_code == "ambiguous_mro"
    assert "unknown.Parent" in review.reason


def test_structural_stdlib_bases_do_not_hide_verified_overrides(
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
from typing import Protocol


class AbstractBase(ABC):
    def hook(self):
        pass


class Interface(Protocol):
    def protocol_hook(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.model import AbstractBase
from vllm.model import Interface


class Child(AbstractBase, Interface):
    def hook(self):
        pass

    def protocol_hook(self):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    endpoints = {
        (relation.downstream_owner, relation.downstream_name)
        for relation in relations
        if relation.relation == "override"
    }
    assert ("Child", "hook") in endpoints
    assert ("Child", "protocol_hook") in endpoints
    assert not [finding for finding in findings if finding.reason_code == "ambiguous_mro"]


def test_external_source_sha_must_match_git_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "external-repo"
    external_root.mkdir()
    monkeypatch.setattr(generator, "_git_head", lambda root: "actual-sha")

    assert generator._verified_external_sources(
        {"external": external_root},
        {"external": "actual-sha"},
    ) == {"external": "actual-sha"}
    with pytest.raises(SystemExit, match="SHA mismatch"):
        generator._verified_external_sources(
            {"external": external_root},
            {"external": "claimed-sha"},
        )


def test_external_source_snapshot_verifies_every_python_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "external-repo"
    source = external_root / "external" / "module.py"
    _write(external_root, "external/module.py", "class ExternalBase:\n    pass\n")
    digest = generator.hashlib.sha256(source.read_bytes()).hexdigest()
    (external_root / ".interface-source.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "package": "external",
                "repository": "https://example.invalid/external",
                "commit": "source-commit",
                "files": {"external/module.py": digest},
            }
        ),
        encoding="utf-8",
    )

    def no_git_checkout(root: Path) -> str:
        raise generator.subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(generator, "_git_head", no_git_checkout)
    assert generator._verified_external_sources(
        {"external": external_root},
        {"external": "source-commit"},
    ) == {"external": "source-commit"}

    source.write_text("class Changed:\n    pass\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="digest mismatch"):
        generator._verified_external_sources(
            {"external": external_root},
            {"external": "source-commit"},
        )


def test_patch_scanner_resolves_local_imports_aliases_and_evidence(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/core.py",
        """
class Engine:
    def step(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/implementation.py",
        """
def imported_patch(self, value):
    return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.core import Engine as ImportedEngine
from vllm_ascend.implementation import imported_patch

PATCH_TARGET = ImportedEngine
if use_fast_path:
    PATCH_TARGET.step = imported_patch
else:
    PATCH_TARGET.step = imported_patch


def install_patch():
    from vllm.core import Engine

    def local_patch(self, value):
        return value

    Engine.step = local_patch
""",
    )

    relations, unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 2
    imported = next(relation for relation in patches if relation.downstream_name == "imported_patch")
    assert imported.downstream_file == "vllm_ascend/implementation.py"
    assert len(imported.evidence) == 2
    assert {evidence.guards for evidence in imported.evidence} == {
        ("use_fast_path",),
        ("not (use_fast_path)",),
    }

    local = next(relation for relation in patches if relation.downstream_name == "local_patch")
    assert local.downstream_file == "vllm_ascend/plugin.py"
    assert local.evidence[0].scope == "install_patch"
    assert not unresolved


def test_private_helper_owner_emits_each_exact_main_call_binding(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/hunyuan.py",
        """
class ProcessingInfo:
    def load(self, **kwargs):
        return kwargs
""",
    )
    _write(
        vllm_root,
        "vllm/other.py",
        """
class ProcessingInfo:
    def load(self, **kwargs):
        return kwargs
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm_ascend.utils import vllm_version_is


def _patch_processor(owner):
    def replacement(self, **kwargs):
        return kwargs

    owner.ProcessingInfo.load = replacement


def _release_only(owner):
    def release_replacement(self, **kwargs):
        return kwargs

    owner.ProcessingInfo.load = release_replacement


def _ambiguous_owner(owner):
    def ambiguous_replacement(self, **kwargs):
        return kwargs

    owner.ProcessingInfo.load = ambiguous_replacement


def _reassigned_owner(owner):
    owner = passthrough(owner)

    def reassigned_replacement(self, **kwargs):
        return kwargs

    owner.ProcessingInfo.load = reassigned_replacement


def install():
    from vllm import hunyuan as main_hunyuan
    from vllm import other

    _patch_processor(main_hunyuan)
    _ambiguous_owner(main_hunyuan)
    _ambiguous_owner(other)
    _reassigned_owner(main_hunyuan)
    if vllm_version_is("0.25.1"):
        _release_only(main_hunyuan)


def install_local_shadow():
    from vllm import other

    def _patch_processor(owner):
        def local_replacement(self, **kwargs):
            return kwargs

        owner.ProcessingInfo.load = local_replacement

    _patch_processor(other)
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/utils.py",
        "def vllm_version_is(_version):\n    return False\n",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 3
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/hunyuan.py", "ProcessingInfo", "load", "replacement"),
        (
            "vllm/hunyuan.py",
            "ProcessingInfo",
            "load",
            "ambiguous_replacement",
        ),
        (
            "vllm/other.py",
            "ProcessingInfo",
            "load",
            "ambiguous_replacement",
        ),
    }
    assert not findings


def test_literal_sys_modules_binding_preserves_cached_import_target(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/core.py",
        "def build(value):\n    return value\n",
    )
    _write(
        vllm_root,
        "vllm/consumer.py",
        "from vllm.core import build\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import sys


def replacement(value):
    return value


cached = sys.modules.get("vllm.consumer")
if cached is not None:
    cached.build = replacement

required = sys.modules["vllm.consumer"]
required.build = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/core.py"
    assert patches[0].upstream_name == "build"
    assert len(patches[0].evidence) == 2
    assert {evidence.target_expression for evidence in patches[0].evidence} == {"vllm.consumer.build"}
    assert not findings


def test_function_parameters_shadow_outer_module_and_runtime_module_bindings(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/owner_module.py",
        "class ProcessingInfo:\n    def load(self):\n        pass\n",
    )
    _write(
        vllm_root,
        "vllm/core.py",
        "def build(value):\n    return value\n",
    )
    _write(
        vllm_root,
        "vllm/consumer.py",
        "from vllm.core import build\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import sys
import vllm.owner_module as owner


def replacement_load(self):
    pass


def replacement_build(value):
    return value


cached = sys.modules.get("vllm.consumer")


def _patch_owner(target):
    target.ProcessingInfo.load = replacement_load


def install(owner, cached):
    _patch_owner(owner)
    cached.build = replacement_build
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not patches


def test_private_helper_call_in_short_circuited_tag_condition_is_inactive(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/owner_module.py",
        "class ProcessingInfo:\n    def load(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/utils.py",
        "def vllm_version_is(_version):\n    return False\n",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import owner_module
from vllm_ascend.utils import vllm_version_is


def _patch_owner(owner):
    def replacement(self):
        pass

    owner.ProcessingInfo.load = replacement


if vllm_version_is("0.25.1") and _patch_owner(owner_module):
    pass
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not patches


def test_private_helper_called_with_multiple_exact_owners_emits_each_relation(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    for module in ("first", "second"):
        _write(
            vllm_root,
            f"vllm/{module}.py",
            "class ProcessingInfo:\n    def load(self):\n        pass\n",
        )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second


def _patch_owner(owner):
    def replacement(self):
        pass

    owner.ProcessingInfo.load = replacement


def install():
    _patch_owner(first)
    _patch_owner(second)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 2
    assert {relation.upstream_file for relation in patches} == {
        "vllm/first.py",
        "vllm/second.py",
    }
    assert all(relation.upstream_name == "load" for relation in patches)


def test_direct_literal_sys_modules_patch_target_is_resolved(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/core.py",
        "def build(value):\n    return value\n",
    )
    _write(
        vllm_root,
        "vllm/consumer.py",
        "from vllm.core import build\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import sys


def replacement(value):
    return value


sys.modules["vllm.consumer"].build = replacement
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/core.py"
    assert patches[0].upstream_name == "build"
    assert patches[0].evidence[0].target_expression == "vllm.consumer.build"


def test_branch_join_preserves_unknown_parameter_tombstone(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class ProcessingInfo:\n    def load(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def _patch_owner(owner):
    owner.ProcessingInfo.load = replacement


def install(owner, enabled):
    if enabled:
        owner = first
    _patch_owner(owner)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not patches


def test_redefined_private_helpers_keep_call_contexts_separate(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = """
class A:
    def run(self):
        pass


class B:
    def run(self):
        pass
"""
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second


def replace_a(self):
    pass


def replace_b(self):
    pass


def _patch_owner(owner):
    owner.A.run = replace_a


_patch_owner(first)


def _patch_owner(owner):
    owner.B.run = replace_b


_patch_owner(second)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in relations
        if relation.relation == "monkey_patch"
    }
    assert patches == {
        ("vllm/first.py", "A", "run", "replace_a"),
        ("vllm/second.py", "B", "run", "replace_b"),
    }


def test_private_helper_owner_resolves_main_ifexp_and_boolop(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class ProcessingInfo:\n    def load(self):\n        pass\n"
    for module in ("first", "second", "third", "fourth"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/utils.py",
        "def vllm_version_is(_version):\n    return False\n",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, fourth, second, third
from vllm_ascend.utils import vllm_version_is


def replacement(self):
    pass


def _patch_owner(owner):
    owner.ProcessingInfo.load = replacement


def install():
    _patch_owner(first if vllm_version_is("0.25.1") else second)
    _patch_owner(vllm_version_is("0.25.1") and third or fourth)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {relation.upstream_file for relation in patches} == {
        "vllm/second.py",
        "vllm/fourth.py",
    }
    assert all(relation.upstream_name == "load" for relation in patches)


def test_multi_level_direct_literal_sys_modules_patch_target_is_resolved(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/consumer.py",
        """
class Service:
    def build(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import sys


def replacement(self, value):
    return value


sys.modules["vllm.consumer"].Service.build = replacement
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/consumer.py"
    assert patches[0].upstream_owner == "Service"
    assert patches[0].upstream_name == "build"
    assert patches[0].evidence[0].target_expression == "vllm.consumer.Service.build"


def test_private_helper_forwarding_propagates_exact_owner_context(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class ProcessingInfo:\n    def load(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def _inner(owner):
    owner.ProcessingInfo.load = replacement


def _outer(owner):
    _inner(owner)


def install():
    _outer(first)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/first.py"
    assert patches[0].upstream_owner == "ProcessingInfo"
    assert patches[0].upstream_name == "load"


def test_constant_bool_short_circuit_selects_only_active_helper_calls(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class ProcessingInfo:\n    def load(self):\n        pass\n"
    for module in ("first", "second", "third", "fourth"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, fourth, second, third


def replacement(self):
    pass


def _patch_owner(owner):
    owner.ProcessingInfo.load = replacement


def install():
    False and _patch_owner(first)
    True or _patch_owner(second)
    False or _patch_owner(third)
    True and _patch_owner(fourth)
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {relation.upstream_file for relation in patches} == {
        "vllm/third.py",
        "vllm/fourth.py",
    }
    assert all(relation.upstream_name == "load" for relation in patches)


def test_deleted_upstream_helper_owners_are_reported_as_risks(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/present.py",
        "class ExistingService:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import present, removed_module


def replacement(self):
    pass


def _patch_module(owner):
    owner.removed = replacement


def _patch_class(owner):
    owner.run = replacement


_patch_module(removed_module)
_patch_class(present.RemovedService)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not patches
    by_target = {finding.target_expression: finding for finding in findings}
    assert set(by_target) == {
        "vllm.present.RemovedService.run",
        "vllm.removed_module.removed",
    }
    assert all(finding.status == "risk" for finding in by_target.values())
    assert all(not finding.generator_issue for finding in by_target.values())


def test_exact_main_branch_return_stops_following_patch_scan(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def after_tag_return(self):
        pass

    def after_true_return(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/utils.py",
        "def vllm_version_is(_version):\n    return False\n",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target
from vllm_ascend.utils import vllm_version_is


def replacement(self):
    pass


def install_for_main():
    if vllm_version_is("0.25.1"):
        pass
    else:
        return
    Target.after_tag_return = replacement


def install_for_true():
    if True:
        return
    Target.after_true_return = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_try_finally_uses_state_before_return_and_stops_after_try(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second", "dead"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import dead, first, second


def finally_replacement(self):
    pass


def dead_replacement(self):
    pass


def install():
    owner = first
    try:
        owner = second
        return
    finally:
        owner.Target.run = finally_replacement
    dead.Target.run = dead_replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in relations
        if relation.relation == "monkey_patch"
    }
    assert patches == {
        ("vllm/second.py", "Target", "run", "finally_replacement"),
    }
    assert not findings


def test_equivalent_guard_contradictions_do_not_create_false_edges(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second", "third", "fourth"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, fourth, second, third


def replacement(self):
    pass


def _patch_owner(owner):
    owner.Target.run = replacement


def install(enabled, value):
    if enabled:
        _patch_owner(first if not enabled else second)
    if value is None:
        _patch_owner(third if value is not None else fourth)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {relation.upstream_file for relation in patches} == {
        "vllm/second.py",
        "vllm/fourth.py",
    }
    assert not findings


def test_exact_hasattr_true_selects_literal_setattr_member(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def run(self):
        pass

    def fallback(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


if hasattr(Target, "run"):
    selected_name = "run"
else:
    selected_name = "fallback"
setattr(Target, selected_name, replacement)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/base.py"
    assert patches[0].upstream_owner == "Target"
    assert patches[0].upstream_name == "run"
    assert patches[0].downstream_name == "replacement"
    assert not findings


def test_exact_try_import_and_non_none_guard_restore_patch_target(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
try:
    from vllm.first import Target as Selected
except ImportError:
    Selected = None


def replacement(self):
    pass


if Selected is not None:
    Selected.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/first.py"
    assert patches[0].upstream_owner == "Target"
    assert patches[0].upstream_name == "run"
    assert patches[0].downstream_name == "replacement"
    assert not findings


def test_cross_scope_same_name_guards_do_not_prune_reachable_helper_patch(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import base


def replacement(self):
    pass


def _inner(owner, enabled):
    if enabled:
        owner.Target.run = replacement


def install(enabled):
    if not enabled:
        _inner(base, True)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/base.py", "Target", "run", "replacement"),
    }
    assert not findings


def test_compound_non_none_guard_narrows_exact_optional_import(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
try:
    from vllm.base import Target as Selected
except ImportError:
    Selected = None


def replacement(self):
    pass


def install(enabled):
    if Selected is not None and enabled:
        Selected.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/base.py", "Target", "run", "replacement"),
    }
    assert not findings


def test_hasattr_does_not_prove_callable_defined_under_unknown_guard(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


if os.getenv("ENABLE_OPTIONAL"):
    def optional():
        pass


def run():
    pass


def fallback():
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement():
    pass


if hasattr(base, "optional"):
    base.run = replacement
else:
    base.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 2
    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/base.py", None, "fallback", "replacement"),
        ("vllm/base.py", None, "run", "replacement"),
    }
    assert all(evidence.guards for relation in patches for evidence in relation.evidence)
    assert not findings


def test_hasattr_does_not_treat_unexported_child_module_as_package_member(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/child.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def run(self):
        pass

    def fallback(self):
        pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm
from vllm.base import Target


def replacement(self):
    pass


if hasattr(vllm, "child"):
    Target.run = replacement
else:
    Target.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 2
    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/base.py", "Target", "fallback", "replacement"),
        ("vllm/base.py", "Target", "run", "replacement"),
    }
    assert all(evidence.guards for relation in patches for evidence in relation.evidence)
    assert not findings


def test_negative_hasattr_for_different_member_does_not_hide_stale_patch(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Target:\n    pass\n")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


if not hasattr(Target, "a"):
    Target.b = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    assert findings[0].target_expression == "vllm.base.Target.b"
    assert findings[0].status == "risk"
    assert findings[0].reason_code == "possible_stale_patch"
    assert not findings[0].generator_issue


def test_matching_except_uses_state_at_explicit_raise(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second


def replacement(self):
    pass


def install():
    owner = first
    try:
        owner = second
        raise ValueError()
    except ValueError:
        owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in patches
    } == {
        ("vllm/second.py", "Target", "run", "replacement"),
    }
    assert not findings


def test_with_return_stops_following_patch_scan(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/dead.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from contextlib import nullcontext

from vllm import dead


def replacement(self):
    pass


def install():
    with nullcontext():
        return
    dead.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert not findings


def test_unknown_reassignment_tombstones_old_import_owner(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first as owner


def replacement(self):
    pass


def make_dynamic():
    return object()


owner = make_dynamic()
owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    assert findings[0].status == "review"
    assert findings[0].reason_code == "dynamic_patch_owner"
    assert not findings[0].generator_issue


def test_implicit_exception_uses_current_state_and_first_matching_handler(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second", "third"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second, third


def replacement(self):
    pass


def install(external_call):
    owner = first
    try:
        owner = second
        external_call()
    except Exception:
        owner.Target.run = replacement
    except KeyError:
        third.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.relation,
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "monkey_patch",
        "vllm/second.py",
        "Target",
        "run",
        "replacement",
    )
    guards = {guard for evidence in relation.evidence for guard in evidence.guards}
    assert "except Exception" in guards
    assert "except KeyError" not in guards
    assert not findings


def test_implicit_and_explicit_try_exits_reach_their_matching_handlers(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = """
class Target:
    def run(self):
        pass

    def fallback(self):
        pass
"""
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second


def replacement(self):
    pass


def install(may_raise):
    owner = first
    try:
        owner = second
        may_raise()
        raise ValueError()
    except TypeError:
        owner.Target.run = replacement
    except ValueError:
        owner.Target.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 2
    assert {
        (
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_name,
        )
        for relation in relations
    } == {
        ("vllm/second.py", "Target", "fallback", "replacement"),
        ("vllm/second.py", "Target", "run", "replacement"),
    }
    guards_by_member = {
        relation.upstream_name: {guard for evidence in relation.evidence for guard in evidence.guards}
        for relation in relations
    }
    assert "except TypeError" in guards_by_member["run"]
    assert "except ValueError" in guards_by_member["fallback"]
    assert not findings


def test_explicit_raise_matches_builtin_exception_base_before_exception(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("second", "third"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import second, third


def replace_os_error(self):
    pass


def replace_exception(self):
    pass


def install():
    owner = second
    try:
        raise FileNotFoundError()
    except OSError:
        owner.Target.run = replace_os_error
    except Exception:
        third.Target.run = replace_exception
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/second.py",
        "Target",
        "run",
        "replace_os_error",
    )
    guards = {guard for evidence in relation.evidence for guard in evidence.guards}
    assert "except OSError" in guards
    assert "except Exception" not in guards
    assert not findings


def test_conditional_class_method_remains_a_guarded_patch_target(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


class Target:
    if os.getenv("ENABLE_OPTIONAL"):
        def optional(self):
            pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


if hasattr(Target, "optional"):
    Target.optional = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.relation,
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "monkey_patch",
        "vllm/base.py",
        "Target",
        "optional",
        "replacement",
    )
    guards = {guard for evidence in relation.evidence for guard in evidence.guards}
    assert any("hasattr" in guard and "optional" in guard for guard in guards)
    assert not findings


def test_with_nullcontext_as_owner_updates_state_after_with(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from contextlib import nullcontext

from vllm import first, second


def replacement(self):
    pass


def install():
    owner = first
    with nullcontext(second) as owner:
        pass
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/second.py",
        "Target",
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_dynamic_with_owner_tombstones_old_import_inside_body(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install(factory):
    owner = first
    with factory() as owner:
        owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    finding = findings[0]
    assert finding.target_expression == "vllm.first.Target.run"
    assert finding.status == "review"
    assert finding.reason_code == "dynamic_patch_owner"
    assert not finding.generator_issue
    assert any("with-context" in guard for guard in finding.evidence_guards)


def test_unknown_owner_setattr_emits_dynamic_owner_review(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class Target:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install(make_dynamic):
    owner = first
    owner = make_dynamic()
    setattr(owner.Target, "run", replacement)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    finding = findings[0]
    assert finding.target_expression == "vllm.first.Target.run"
    assert finding.status == "review"
    assert finding.reason_code == "dynamic_patch_owner"
    assert not finding.generator_issue


def test_symbol_defined_in_both_unknown_if_arms_proves_hasattr_true(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


if os.getenv("ENABLE_OPTIONAL"):
    def optional():
        pass
else:
    def optional():
        pass


def run():
    pass


def fallback():
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement():
    pass


if hasattr(base, "optional"):
    base.run = replacement
else:
    base.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/base.py",
        None,
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_symbol_defined_in_try_finally_proves_hasattr_true(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
try:
    pass
finally:
    def optional():
        pass


def run():
    pass


def fallback():
    pass
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement():
    pass


if hasattr(base, "optional"):
    base.run = replacement
else:
    base.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/base.py",
        None,
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_suppress_restores_live_state_after_matching_raise(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from contextlib import suppress

from vllm import first, second


def replacement(self):
    pass


def install():
    owner = first
    with suppress(ValueError):
        owner = second
        raise ValueError()
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/second.py",
        "Target",
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_latest_exact_owner_replaces_stale_dynamic_provenance(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    upstream_source = "class Target:\n    def run(self):\n        pass\n"
    for module in ("first", "second"):
        _write(vllm_root, f"vllm/{module}.py", upstream_source)
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second


def replacement(self):
    pass


def install(make_dynamic):
    owner = first
    owner = make_dynamic()
    owner = second
    owner = make_dynamic()
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    finding = findings[0]
    assert finding.target_expression == "vllm.second.Target.run"
    assert finding.status == "review"
    assert finding.reason_code == "dynamic_patch_owner"
    assert not finding.generator_issue


def test_patch_scanner_reports_ambiguous_and_unsupported_patches(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/first.py",
        "class First:\n    def run(self):\n        pass\n",
    )
    _write(
        vllm_root,
        "vllm/second.py",
        "class Second:\n    def run(self):\n        pass\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
try:
    from vllm.first import First as Selected
except ImportError:
    from vllm.second import Second as Selected


def replacement(self):
    pass


Selected.run = replacement

from vllm.first import First
First.run = property(replacement)
""",
    )

    relations, unresolved = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    property_patch = next(relation for relation in relations if relation.relation == "monkey_patch")
    assert property_patch.upstream_owner == "First"
    assert property_patch.upstream_name == "run"
    assert property_patch.downstream_name == "replacement"
    assert property_patch.evidence[0].patch_kind == "property"
    assert any(
        relation.reason == "ambiguous patch target alias"
        and "vllm.first.First.run" in relation.target_expression
        and "vllm.second.Second.run" in relation.target_expression
        for relation in unresolved
    )
    assert not any(relation.reason == "property patch is outside callable mapping scope" for relation in unresolved)


def test_class_body_callable_alias_is_a_method_and_patch_replacement(
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


def alias_hook(self, value):
    return value


class Child(Base):
    hook = alias_hook


Base.hook = Child.hook
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    alias_relations = [
        relation
        for relation in relations
        if relation.downstream_owner == "Child" and relation.downstream_name == "hook"
    ]
    assert {relation.relation for relation in alias_relations} == {
        "monkey_patch",
        "override",
    }
    patch = next(relation for relation in alias_relations if relation.relation == "monkey_patch")
    assert patch.evidence[0].binding_line is not None
    assert patch.evidence[0].definition_line is not None
    assert not findings


def test_wrapper_factory_return_and_local_binding_are_resolved(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def first(self, value):
        return value

    def second(self, value):
        return value

    def third(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def wrap_or_identity(original):
    if getattr(original, "already_wrapped", False):
        return original

    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    return wrapped


def make_exact():
    def exact(self, value):
        return value

    return exact


def ambiguous(flag):
    def first(self, value):
        return value

    def second(self, value):
        return value

    if flag:
        return first
    return second


produced = wrap_or_identity(Target.first)
Target.first = produced
Target.second = make_exact()
Target.third = ambiguous(flag)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = {relation.upstream_name: relation for relation in relations if relation.relation == "monkey_patch"}
    assert set(patches) == {"first", "second"}
    assert patches["first"].downstream_name == "wrapped"
    assert patches["first"].evidence[0].patch_kind == "wrapper_or_identity"
    assert patches["second"].downstream_name == "exact"
    assert patches["second"].evidence[0].patch_kind == "wrapper_factory"
    assert len(findings) == 1
    assert findings[0].reason_code == "ambiguous_wrapper_factory"


def test_save_and_restore_original_are_lifecycle_findings(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def run(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def install():
    original = getattr(Target, "run", None)
    Target._vllm_ascend_original_run = original

    def replacement(self, value):
        return value

    try:
        Target.run = replacement
    finally:
        Target.run = original
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].downstream_name == "replacement"
    assert {finding.reason_code for finding in findings} == {
        "restore_original",
        "save_original",
    }
    assert all(finding.status == "excluded" for finding in findings)
    assert all(not finding.generator_issue for finding in findings)


def test_field_writes_are_classified_without_callable_resolution(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/state.py",
        """
module_flag: bool = True


class State:
    class_flag: bool = True


singleton = State()


def callable_target(value):
    return value


callable_target = callable_target
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import state

state.module_flag = False
state.State.class_flag = False

item = state.singleton
if not hasattr(item, "extra"):
    item.extra = None


def replacement(value):
    return value


state.callable_target = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_name == "callable_target"
    assert len(findings) == 3
    assert sum(finding.status == "verified" and finding.reason_code == "field_mutation" for finding in findings) == 2
    injected = next(finding for finding in findings if finding.reason_code == "inject_missing_field")
    assert injected.status == "expected"
    assert not injected.generator_issue


def test_lambda_patch_and_parse_failures_are_explicit(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/core.py",
        "class Engine:\n    def step(self, value):\n        return value\n",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.core import Engine

Engine.step = lambda self, value: value
""",
    )

    relations, _ = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    patch = next(relation for relation in relations if relation.relation == "monkey_patch")
    assert patch.downstream_name.startswith("<lambda>@")
    assert patch.downstream_signature == [
        "sync",
        [],
        [["self", True], ["value", True]],
        None,
        [],
        None,
    ]
    assert patch.evidence[0].patch_kind == "lambda"

    _write(ascend_root, "vllm_ascend/broken.py", "def broken(:\n")
    with pytest.raises(ValueError, match="Python source parsing failed"):
        generator.InterfaceBoundaryGenerator(vllm_root, ascend_root)


def test_upstream_patch_method_deletion_becomes_a_risk(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/api.py",
        """
class Target:
    def hook(self, value):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.api import Target


def replacement(self, value):
    return value


Target.hook = replacement
""",
    )

    before_relations, before_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    assert any(
        relation.relation == "monkey_patch" and relation.upstream_owner == "Target" and relation.upstream_name == "hook"
        for relation in before_relations
    )
    assert not before_findings

    _write(
        vllm_root,
        "vllm/api.py",
        """
class Target:
    pass
""",
    )
    after_relations, after_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not any(
        relation.relation == "monkey_patch" and relation.upstream_name == "hook" for relation in after_relations
    )
    risk = next(finding for finding in after_findings if finding.target_expression == "vllm.api.Target.hook")
    assert risk.status == "risk"
    assert risk.reason_code == "possible_stale_patch"
    assert not risk.generator_issue


def test_new_downstream_patch_for_missing_upstream_method_is_a_risk(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/api.py", "class Target:\n    pass\n")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        "from vllm.api import Target\n",
    )

    before_relations, before_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    assert not before_relations
    assert not before_findings

    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.api import Target


def added_patch(self, value):
    return value


Target.new_hook = added_patch
""",
    )
    after_relations, after_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not after_relations
    risk = next(finding for finding in after_findings if finding.target_expression == "vllm.api.Target.new_hook")
    assert risk.downstream_name == "added_patch"
    assert risk.status == "risk"
    assert risk.reason_code == "possible_stale_patch"
    assert not risk.generator_issue


def test_upstream_base_deletion_becomes_an_inheritance_risk(
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
    pass
""",
    )

    before_relations, before_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    assert any(
        relation.relation == "inheritance" and relation.upstream_name == "Base" and relation.downstream_owner == "Child"
        for relation in before_relations
    )
    assert not before_findings

    _write(vllm_root, "vllm/base.py", "")
    after_relations, after_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not any(
        relation.relation == "inheritance" and relation.downstream_owner == "Child" for relation in after_relations
    )
    risk = next(
        finding
        for finding in after_findings
        if finding.relation == "inheritance" and finding.downstream_owner == "Child"
    )
    assert risk.target_expression == "vllm.base.Base"
    assert risk.status == "risk"
    assert risk.reason_code == "missing_upstream_base"
    assert not risk.generator_issue


def test_upstream_signature_change_updates_the_existing_relation(
    tmp_path: Path,
) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(
        vllm_root,
        "vllm/api.py",
        """
class Target:
    def hook(self, value, *, mode=None):
        return value
""",
    )
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.api import Target


def replacement(self, *args, **kwargs):
    return None


Target.hook = replacement
""",
    )

    before_relations, before_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    before = next(
        relation
        for relation in before_relations
        if relation.relation == "monkey_patch" and relation.upstream_name == "hook"
    )
    assert before.upstream_signature == [
        "sync",
        [],
        [["self", True], ["value", True]],
        None,
        [["mode", False]],
        None,
    ]
    assert not before_findings

    _write(
        vllm_root,
        "vllm/api.py",
        """
class Target:
    def hook(self, value, context, *, mode=None):
        return value
""",
    )
    after_relations, after_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()
    after = next(
        relation
        for relation in after_relations
        if relation.relation == "monkey_patch" and relation.upstream_name == "hook"
    )

    assert after.downstream_key() == before.downstream_key()
    assert after.upstream_signature == [
        "sync",
        [],
        [["self", True], ["value", True], ["context", True]],
        None,
        [["mode", False]],
        None,
    ]
    assert after.upstream_signature != before.upstream_signature
    assert not after_findings


def _v018_source_roots(tmp_path: Path) -> tuple[Path, Path]:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    return vllm_root, ascend_root


def _write_run_target(vllm_root: Path, module: str) -> None:
    _write(
        vllm_root,
        f"vllm/{module}.py",
        "class Target:\n    def run(self):\n        pass\n",
    )


def test_proven_safe_call_does_not_activate_exception_handler_patch(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install():
    try:
        len(())
    except ValueError:
        first.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert not findings


def test_conditional_callable_variants_are_reported_instead_of_arbitrarily_selected(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


class Target:
    if os.getenv("USE_SHORT_SIGNATURE"):
        def run(self, value):
            pass
    else:
        def run(self, value, context):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    pass


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_file,
        finding.downstream_owner,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
        finding.evidence_guards,
    ) == (
        "monkey_patch",
        "vllm_ascend/plugin.py",
        None,
        "replacement",
        "vllm.base.Target.run",
        "review",
        "conditional_callable_variants",
        False,
        (),
    )


def test_override_tracks_conditional_owner_and_unconditional_ancestor(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


class Root:
    def run(self, root_value):
        pass


class Base(Root):
    if os.getenv("ENABLE_BASE_RUN"):
        def run(self, base_value, context):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 3
    assert {
        (
            relation.relation,
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_owner,
            relation.downstream_name,
        )
        for relation in relations
    } == {
        (
            "inheritance",
            "vllm/base.py",
            None,
            "Base",
            "Child",
            "Base",
        ),
        (
            "override",
            "vllm/base.py",
            "Base",
            "run",
            "Child",
            "run",
        ),
        (
            "override",
            "vllm/base.py",
            "Root",
            "run",
            "Child",
            "run",
        ),
    }
    assert not findings


def test_terminating_if_arm_does_not_make_module_symbol_optional(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


if os.getenv("FAIL_IMPORT"):
    raise RuntimeError()
else:
    def optional():
        pass


def run():
    pass


def fallback():
    pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement():
    pass


if hasattr(base, "optional"):
    base.run = replacement
else:
    base.fallback = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.relation,
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "monkey_patch",
        "vllm/base.py",
        None,
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_unreachable_conditional_method_definition_is_not_indexed(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
import os


class Target:
    if os.getenv("USE_TARGET"):
        def run(self):
            pass
    else:
        raise RuntimeError()

        def run(self, unreachable):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.upstream_signature,
        relation.downstream_name,
    ) == (
        "vllm/base.py",
        "Target",
        "run",
        ["sync", [], [["self", True]], None, [], None],
        "replacement",
    )
    assert not findings


def test_suppress_in_multiple_context_managers_restores_live_state(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    for module in ("first", "second"):
        _write_run_target(vllm_root, module)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from contextlib import nullcontext, suppress

from vllm import first, second


def replacement(self):
    pass


def install():
    owner = first
    with nullcontext(), suppress(ValueError):
        owner = second
        raise ValueError()
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.relation,
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "monkey_patch",
        "vllm/second.py",
        "Target",
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_local_exception_does_not_match_same_named_builtin_handler(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    for module in ("first", "second"):
        _write_run_target(vllm_root, module)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import builtins

from vllm import first, second


class ValueError(Exception):
    pass


def replacement(self):
    pass


def install():
    owner = first
    try:
        raise ValueError()
    except builtins.ValueError:
        owner = first
    except Exception:
        owner = second
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/second.py",
        "Target",
        "run",
        "replacement",
    )
    assert all(not evidence.guards for evidence in relation.evidence)
    assert not findings


def test_tuple_exception_handler_preserves_its_real_evidence(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install():
    try:
        raise FileNotFoundError()
    except (OSError, ValueError):
        first.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len(relations) == 1
    relation = relations[0]
    assert (
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        relation.downstream_name,
    ) == (
        "vllm/first.py",
        "Target",
        "run",
        "replacement",
    )
    assert {guard for evidence in relation.evidence for guard in evidence.guards} == {"except (OSError, ValueError)"}
    assert not findings


def test_none_refinement_clears_stale_upstream_provenance(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def factory():
    return object()


def replacement(self):
    pass


def install(flag):
    owner = first
    if flag:
        owner = first
    else:
        owner = None
    if owner is None:
        owner = factory()
        owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not relations
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_file,
        finding.downstream_owner,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
        finding.evidence_guards,
    ) == (
        "monkey_patch",
        "vllm_ascend/plugin.py",
        None,
        "replacement",
        "owner.Target.run",
        "review",
        "dynamic_patch_owner",
        False,
        ("owner is None",),
    )
    assert "vllm.first" not in finding.target_expression


def _v019_run_signature(
    *positional: tuple[str, bool],
    keyword_only: tuple[tuple[str, bool], ...] = (),
) -> list[object]:
    return [
        "sync",
        [],
        [[name, required] for name, required in positional],
        None,
        [[name, required] for name, required in keyword_only],
        None,
    ]


def _v019_relation_payload(relations: list[generator.Relation]) -> list[tuple[object, ...]]:
    return [
        (
            relation.relation,
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            relation.upstream_signature,
            relation.downstream_file,
            relation.downstream_owner,
            relation.downstream_name,
            relation.downstream_signature,
            tuple(evidence.as_dict().items() for evidence in relation.evidence),
        )
        for relation in relations
    ]


def test_v019_later_unconditional_method_replaces_conditional_definition(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    if feature_enabled():
        def run(self, legacy_value):
            pass

    def run(self, value, *, context=None):
        pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


def replacement(self, value, *, context=None):
    pass


Base.run = replacement


class Child(Base):
    def run(self, value, *, context=None):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    method_relations = [relation for relation in relations if relation.relation in {"monkey_patch", "override"}]
    assert {
        (
            relation.relation,
            relation.upstream_owner,
            relation.downstream_owner,
            relation.downstream_name,
        )
        for relation in method_relations
    } == {
        ("monkey_patch", "Base", None, "replacement"),
        ("override", "Base", "Child", "run"),
    }
    expected = _v019_run_signature(
        ("self", True),
        ("value", True),
        keyword_only=(("context", False),),
    )
    assert all(relation.upstream_signature == expected for relation in method_relations)
    assert not findings


def test_v019_overload_stubs_do_not_replace_runtime_implementation(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
from typing import overload


class Base:
    @overload
    def run(self, value: int) -> int: ...

    @overload
    def run(self, value: str, *, strict: bool = False) -> str: ...

    def run(self, value, *, context=None):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


def replacement(self, value, *, context=None):
    return value


Base.run = replacement


class Child(Base):
    def run(self, value, *, context=None):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    method_relations = [relation for relation in relations if relation.relation in {"monkey_patch", "override"}]
    assert len(method_relations) == 2
    expected = _v019_run_signature(
        ("self", True),
        ("value", True),
        keyword_only=(("context", False),),
    )
    assert all(relation.upstream_signature == expected for relation in method_relations)
    assert not findings


def test_v019_same_signature_branch_order_has_identical_output(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


Target.run = replacement
""",
    )

    def generate(if_result: str, else_result: str) -> tuple[list[object], list[object]]:
        _write(
            vllm_root,
            "vllm/base.py",
            f'''\
class Target:
    if feature_enabled():
        def run(self, value):
            return "{if_result}"
    else:
        def run(self, value):
            return "{else_result}"
''',
        )
        return generator.InterfaceBoundaryGenerator(
            vllm_root,
            ascend_root,
        ).generate()

    first_relations, first_findings = generate("first", "second")
    second_relations, second_findings = generate("second", "first")

    first_patches = [relation for relation in first_relations if relation.relation == "monkey_patch"]
    second_patches = [relation for relation in second_relations if relation.relation == "monkey_patch"]
    assert len(first_patches) == len(second_patches) == 1
    assert _v019_relation_payload(first_relations) == _v019_relation_payload(
        second_relations,
    )
    assert first_findings == second_findings == []


def test_v019_conditional_override_signature_variants_are_reviewed(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    if feature_enabled():
        def run(self, value):
            pass
    else:
        def run(self, value, context):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "override"]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_owner,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "override",
        "Child",
        "run",
        "vllm.base.Base.run",
        "review",
        "conditional_callable_variants",
        False,
    )


def test_v019_top_level_conditional_same_signature_is_one_relation(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    def run(value):
        return value
else:
    def run(value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_signature == _v019_run_signature(("value", True))
    assert not findings


def test_v019_top_level_conditional_signature_variants_are_reviewed(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    def run(value):
        return value
else:
    def run(value, context):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "monkey_patch",
        "replacement",
        "vllm.base.run",
        "review",
        "conditional_callable_variants",
        False,
    )


def test_v019_later_non_callable_or_delete_removes_old_method_target(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class NonCallableTarget:
    def run(self, value):
        pass

    run = None


class DeletedTarget:
    def run(self, value):
        pass

    del run
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import DeletedTarget, NonCallableTarget


def replacement(self, value):
    pass


NonCallableTarget.run = replacement
DeletedTarget.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            finding.target_expression,
            finding.status,
            finding.reason_code,
            finding.generator_issue,
        )
        for finding in findings
    } == {
        (
            "vllm.base.NonCallableTarget.run",
            "risk",
            "possible_stale_patch",
            False,
        ),
        (
            "vllm.base.DeletedTarget.run",
            "risk",
            "possible_stale_patch",
            False,
        ),
    }


def test_v019_terminating_definition_path_does_not_contribute_variant(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def run(self, dead_value):
            pass
        raise RuntimeError()
    else:
        def run(self, value, *, context=None):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value, *, context=None):
    pass


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_signature == _v019_run_signature(
        ("self", True),
        ("value", True),
        keyword_only=(("context", False),),
    )
    assert not findings


def test_v019_finally_definition_is_the_final_patch_and_override_binding(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    try:
        def run(self, try_value):
            pass
        may_raise()
    except ImportError:
        def run(self, except_value, extra):
            pass
    finally:
        def run(self, value, *, context=None):
            pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


def replacement(self, value, *, context=None):
    pass


Base.run = replacement


class Child(Base):
    def run(self, value, *, context=None):
        pass
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    method_relations = [relation for relation in relations if relation.relation in {"monkey_patch", "override"}]
    assert len(method_relations) == 2
    expected = _v019_run_signature(
        ("self", True),
        ("value", True),
        keyword_only=(("context", False),),
    )
    assert all(relation.upstream_signature == expected for relation in method_relations)
    assert not findings


def test_v019_method_after_raising_finally_is_unreachable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    try:
        pass
    finally:
        raise RuntimeError()

    def run(self, value):
        pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(self, value):
    pass


base.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "vllm.base.Target.run",
        "risk",
        "possible_stale_patch",
        False,
    )


def test_v019_function_local_late_len_binding_makes_handler_reachable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install():
    try:
        len(())
    except UnboundLocalError:
        first.Target.run = replacement
    len = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert {guard for item in patches[0].evidence for guard in item.guards} == {
        "except UnboundLocalError",
    }
    assert not findings


def test_v019_module_late_len_definition_keeps_earlier_builtin_call_safe(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


try:
    len(())
except UnboundLocalError:
    first.Target.run = replacement


def len(value):
    return 0
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v019_partially_unknown_tuple_handler_keeps_possible_path(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


def replacement(self):
    pass


def install(errors):
    try:
        raise FileNotFoundError()
    except (ValueError, errors.UnknownError):
        first.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert {guard for item in patches[0].evidence for guard in item.guards} == {
        "except (ValueError, errors.UnknownError)",
    }
    assert not findings


def test_v019_function_local_dynamic_owner_does_not_inherit_module_provenance(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first


owner = first


def factory():
    return object()


def replacement(self):
    pass


def install():
    owner = factory()
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v019_empty_nullcontext_and_suppress_add_no_guard(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "first")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from contextlib import nullcontext, suppress

from vllm import first


def replacement(self):
    pass


def install():
    with nullcontext(), suppress():
        first.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert all(not evidence.guards for evidence in patches[0].evidence)
    assert not findings


def _v020_assert_conditional_callable_presence(
    relations: list[generator.Relation],
    findings: list[generator.UnresolvedRelation],
    *,
    target_expression: str,
    downstream_owner: str | None = None,
) -> None:
    assert not [relation for relation in relations if relation.relation in {"monkey_patch", "override"}]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_owner,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "override" if downstream_owner else "monkey_patch",
        downstream_owner,
        "run" if downstream_owner else "replacement",
        target_expression,
        "review",
        "conditional_callable_presence",
        False,
    )


def test_v020_class_callable_or_none_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def run(self, value):
            return value
    else:
        run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.Target.run",
    )


def test_v020_module_callable_or_none_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    def run(value):
        return value
else:
    run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.run",
    )


def test_v020_class_callable_or_absent_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def run(self, value):
            return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.Target.run",
    )


def test_v020_module_callable_or_absent_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    def run(value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.run",
    )


def test_v020_exact_hasattr_selects_callable_from_absent_module_path(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    def run(value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


if hasattr(base, "run"):
    base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_file,
        patches[0].upstream_owner,
        patches[0].upstream_name,
        patches[0].upstream_signature,
    ) == (
        "vllm/base.py",
        None,
        "run",
        _v019_run_signature(("value", True)),
    )
    assert any("hasattr" in guard and "run" in guard for evidence in patches[0].evidence for guard in evidence.guards)
    assert not findings


def test_v020_exact_hasattr_does_not_prove_callable_against_none(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def run(self, value):
            return value
    else:
        run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


if hasattr(Target, "run"):
    Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.Target.run",
    )
    assert not findings[0].evidence_guards


def test_v020_none_shadow_in_mro_does_not_invent_root_override(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Root:
    def run(self, root_value):
        return root_value


class Base(Root):
    if feature_enabled():
        def run(self, base_value):
            return base_value
    else:
        run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = [relation for relation in relations if relation.relation == "override"]
    assert not overrides
    _v020_assert_conditional_callable_presence(
        relations,
        findings,
        target_expression="vllm.base.Base.run",
        downstream_owner="Child",
    )


def test_v020_plain_definition_does_not_activate_unreachable_handler_variant(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    try:
        def run(self, value):
            return value
    except ValueError:
        def run(self, value, context):
            return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_signature == _v019_run_signature(
        ("self", True),
        ("value", True),
    )
    assert not findings


def test_v020_unmatched_raise_makes_later_module_endpoints_unreachable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/dead.py",
        """
try:
    raise ValueError()
except ImportError:
    pass


def run(value):
    return value


class Target:
    def run(self, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.dead as dead


def replacement(*args):
    return args


dead.run = replacement
dead.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert {
        (
            finding.target_expression,
            finding.status,
            finding.reason_code,
            finding.generator_issue,
        )
        for finding in findings
    } == {
        (
            "vllm.dead.run",
            "risk",
            "possible_stale_patch",
            False,
        ),
        (
            "vllm.dead.Target.run",
            "risk",
            "possible_stale_patch",
            False,
        ),
    }


def test_v020_try_definition_survives_every_normally_completing_path(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Root:
    def run(self, root_value):
        return root_value


class Base(Root):
    try:
        def run(self, base_value):
            return base_value
        may_raise()
    except Exception:
        pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = [relation for relation in relations if relation.relation == "override"]
    assert len(overrides) == 1
    assert (
        overrides[0].upstream_owner,
        overrides[0].upstream_name,
        overrides[0].upstream_signature,
    ) == (
        "Base",
        "run",
        _v019_run_signature(("self", True), ("base_value", True)),
    )
    assert not findings


def test_v020_handler_uses_state_at_exact_throwing_statement(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    for module in ("first", "second", "third"):
        _write_run_target(vllm_root, module)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, second, third


def replacement(self):
    pass


def install():
    owner = first
    try:
        owner = second
        raise ValueError()
        owner = third
    except ValueError:
        owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_file,
        patches[0].upstream_owner,
        patches[0].upstream_name,
    ) == (
        "vllm/second.py",
        "Target",
        "run",
    )
    assert {guard for evidence in patches[0].evidence for guard in evidence.guards} == {"except ValueError"}
    assert not findings


def test_v020_first_matching_exception_handler_consumes_exact_raise(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    for module in ("first", "second", "initial"):
        _write_run_target(vllm_root, module)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import first, initial, second


def replacement(self):
    pass


def install():
    owner = initial
    try:
        raise FileNotFoundError()
    except OSError:
        owner = first
    except Exception:
        owner = second
    owner.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_file,
        patches[0].upstream_owner,
        patches[0].upstream_name,
    ) == (
        "vllm/first.py",
        "Target",
        "run",
    )
    assert not findings


def test_v021_conditional_imported_callable_alias_stays_callable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/cpu.py",
        """
def run_cpu(value, *, mode=None):
    return value
""",
    )
    _write(
        vllm_root,
        "vllm/base.py",
        """
def run(value, *, mode=None):
    return value


if use_cpu_implementation():
    from vllm.cpu import run_cpu

    run = run_cpu
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value, *, mode=None):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_signature == _v019_run_signature(
        ("value", True),
        keyword_only=(("mode", False),),
    )
    assert not findings


def test_v022_cpu_only_callable_alias_is_inactive_for_npu_mapping(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/cpu.py",
        """
def run_cpu(value, cpu_context, **kwargs):
    return value
""",
    )
    _write(
        vllm_root,
        "vllm/base.py",
        """
def run(value, *, mode=None):
    return value


if current_platform.is_cpu():
    from vllm.cpu import run_cpu

    run = run_cpu
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value, *, mode=None):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_signature == _v019_run_signature(
        ("value", True),
        keyword_only=(("mode", False),),
    )
    assert not findings


def test_v021_annotated_class_callable_alias_is_a_method(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
from collections.abc import Callable


def default_run(self, value):
    return value


class Base:
    run: Callable[..., object] = default_run
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


def replacement(self, value):
    return value


Base.run = replacement


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    method_relations = [relation for relation in relations if relation.relation in {"monkey_patch", "override"}]
    assert {
        (
            relation.relation,
            relation.upstream_owner,
            relation.upstream_name,
            relation.downstream_owner,
            relation.downstream_name,
        )
        for relation in method_relations
    } == {
        ("monkey_patch", "Base", "run", None, "replacement"),
        ("override", "Base", "run", "Child", "run"),
    }
    expected = _v019_run_signature(("self", True), ("value", True))
    assert all(relation.upstream_signature == expected for relation in method_relations)
    assert not findings


def test_v023_assert_false_activates_matching_exception_handler(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "target")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import target


def replacement(self):
    pass


def install():
    try:
        assert False
    except AssertionError:
        target.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_file,
        patches[0].upstream_owner,
        patches[0].upstream_name,
    ) == ("vllm/target.py", "Target", "run")
    assert {guard for evidence in patches[0].evidence for guard in evidence.guards} == {"except AssertionError"}
    assert not findings


def test_v023_short_circuited_false_and_call_cannot_reach_handler(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "target")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import target


def replacement(self):
    pass


def install():
    try:
        False and may_raise()
    except Exception:
        target.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v023_custom_exception_inheritance_selects_first_matching_handler(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    for module in ("lookup", "value", "fallback"):
        _write_run_target(vllm_root, module)
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import fallback, lookup, value


class PluginError(ValueError):
    pass


def replacement(self):
    pass


def install():
    try:
        raise PluginError()
    except LookupError:
        lookup.Target.run = replacement
    except ValueError:
        value.Target.run = replacement
    except Exception:
        fallback.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/value.py"
    assert {guard for evidence in patches[0].evidence for guard in evidence.guards} == {"except ValueError"}
    assert not findings


def test_v023_negative_hasattr_narrows_callable_or_unbound_to_injection(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def run(self, value):
            return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


if not hasattr(Target, "run"):
    Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "vllm.base.Target.run",
        "expected",
        "inject_missing_member",
        False,
    )
    assert any("not hasattr" in guard for guard in finding.evidence_guards)


def test_v023_downstream_method_callable_or_none_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    def run(self, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    if feature_enabled():
        def run(self, value):
            return value
    else:
        run = None
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "override"]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.downstream_owner,
        finding.downstream_name,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "override",
        "Child",
        "run",
        "vllm.base.Base.run",
        "review",
        "conditional_callable_presence",
        False,
    )


def test_v023_upstream_base_class_or_none_is_presence_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    class Base:
        def run(self, value):
            return value
else:
    Base = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation in {"inheritance", "override"}]
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
    ) == (
        "inheritance",
        "vllm.base.Base",
        "review",
        "conditional_class_presence",
        False,
    )


def test_v023_same_name_conditional_classes_keep_both_method_signatures(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
if feature_enabled():
    class Base:
        def run(self, value):
            return value
else:
    class Base:
        def run(self, value, context):
            return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    inheritance = [relation for relation in relations if relation.relation == "inheritance"]
    assert len(inheritance) == 1
    assert not [relation for relation in relations if relation.relation == "override"]
    assert len(findings) == 1
    assert (
        findings[0].relation,
        findings[0].target_expression,
        findings[0].status,
        findings[0].reason_code,
        findings[0].generator_issue,
    ) == (
        "override",
        "vllm.base.Base.run",
        "review",
        "conditional_callable_variants",
        False,
    )


def test_v023_same_class_staticmethod_assignment_is_patchable_interface(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def _run(value, *, mode=None):
        return value

    run = staticmethod(_run)
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(value, *, mode=None):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_owner,
        patches[0].upstream_name,
        patches[0].upstream_signature,
    ) == (
        "Target",
        "run",
        _v019_run_signature(
            ("value", True),
            keyword_only=(("mode", False),),
        ),
    )
    assert not findings


def test_v023_same_class_classmethod_assignment_is_patchable_interface(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    def _run(cls, value):
        return value

    run = classmethod(_run)
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(cls, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_owner,
        patches[0].upstream_name,
        patches[0].upstream_signature,
    ) == (
        "Target",
        "run",
        _v019_run_signature(("cls", True), ("value", True)),
    )
    assert not findings


def test_v023_definite_none_member_is_stale_not_conditional_callable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    assert (
        findings[0].target_expression,
        findings[0].status,
        findings[0].reason_code,
        findings[0].generator_issue,
    ) == (
        "vllm.base.Target.run",
        "risk",
        "possible_stale_patch",
        False,
    )
    assert "some normally completing paths" not in findings[0].reason


def test_v023_missing_upstream_super_method_is_reported_as_risk(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    pass
""",
    )
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

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len([relation for relation in relations if relation.relation == "inheritance"]) == 1
    assert not [relation for relation in relations if relation.relation == "override"]
    assert len(findings) == 1
    assert (
        findings[0].relation,
        findings[0].downstream_owner,
        findings[0].downstream_name,
        findings[0].target_expression,
        findings[0].status,
        findings[0].reason_code,
        findings[0].generator_issue,
    ) == (
        "override",
        "Child",
        "run",
        "vllm.base.Base.run",
        "risk",
        "missing_upstream_super_target",
        False,
    )


def test_v023_negative_hasattr_excludes_unconditionally_present_callable(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write_run_target(vllm_root, "target")
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm import target


def replacement(self):
    pass


if not hasattr(target.Target, "run"):
    target.Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v023_negative_hasattr_excludes_present_none_member(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    run = None
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


if not hasattr(Target, "run"):
    Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v023_negative_hasattr_sees_unconditional_inherited_member(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    def run(self):
        pass


class Target(Base):
    pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    pass


if not hasattr(Target, "run"):
    Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert not findings


def test_v023_wrapper_alias_preserves_conditional_source_presence(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    if feature_enabled():
        def _run(value):
            return value
    else:
        _run = None

    run = staticmethod(_run)
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(findings) == 1
    assert (
        findings[0].target_expression,
        findings[0].status,
        findings[0].reason_code,
        findings[0].generator_issue,
    ) == (
        "vllm.base.Target.run",
        "review",
        "conditional_callable_presence",
        False,
    )


def test_v023_module_level_classmethod_wrapper_is_not_a_callable_endpoint(
    tmp_path: Path,
) -> None:
    vllm_root, _ = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
def implementation(cls, value):
    return value


run = classmethod(implementation)
""",
    )

    index = generator.RepositoryIndex(vllm_root, "vllm")

    assert {binding.kind for binding in index.find_final_bindings("vllm.base.run")} == {"value"}
    assert not index.find_final_callable_variants("vllm.base.run")


def test_v023_dead_super_call_does_not_create_missing_target_risk(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        if False:
            return super().run(value)
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len([relation for relation in relations if relation.relation == "inheritance"]) == 1
    assert not [relation for relation in relations if relation.relation == "override"]
    assert not findings


def test_v023_builtin_object_super_methods_are_not_missing_upstream_targets(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    pass
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return super().__repr__()
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert len([relation for relation in relations if relation.relation == "inheritance"]) == 1
    assert not [relation for relation in relations if relation.relation == "override"]
    assert not findings


def test_v024_records_exact_descriptor_kinds_for_overrides(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    def ordinary(self, value):
        return value

    @property
    def state(self):
        return 1

    @classmethod
    def build(cls, value):
        return cls()

    @staticmethod
    def normalize(value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def ordinary(self, value):
        return value

    @property
    def state(self):
        return 2

    @classmethod
    def build(cls, value):
        return cls()

    @staticmethod
    def normalize(value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = {relation.downstream_name: relation for relation in relations if relation.relation == "override"}
    assert set(overrides) == {"ordinary", "state", "build", "normalize"}
    assert {
        name: (
            relation.upstream_descriptor_kind,
            relation.downstream_descriptor_kind,
            relation.installed_descriptor_kind,
        )
        for name, relation in overrides.items()
    } == {
        "ordinary": ("ordinary", "ordinary", "ordinary"),
        "state": ("property", "property", "property"),
        "build": ("classmethod", "classmethod", "classmethod"),
        "normalize": ("staticmethod", "staticmethod", "staticmethod"),
    }
    assert not findings


def test_v024_override_descriptor_mismatch_keeps_relation_and_supplemental_finding(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    @property
    def state(self):
        return 1
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def state(self):
        return 2
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = [relation for relation in relations if relation.relation == "override"]
    assert len(overrides) == 1
    assert (
        overrides[0].upstream_descriptor_kind,
        overrides[0].downstream_descriptor_kind,
        overrides[0].installed_descriptor_kind,
    ) == ("property", "ordinary", "ordinary")
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.relation,
        finding.target_expression,
        finding.status,
        finding.reason_code,
        finding.generator_issue,
        finding.supplemental,
        finding.upstream_descriptor_kind,
        finding.downstream_descriptor_kind,
        finding.installed_descriptor_kind,
    ) == (
        "override",
        "vllm.base.Base.state",
        "review",
        "descriptor_kind_mismatch",
        False,
        True,
        "property",
        "ordinary",
        "ordinary",
    )


def test_v024_property_patch_keeps_definition_and_installed_kinds_separate(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    @property
    def state(self):
        return 1
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


def replacement(self):
    return 2


Target.state = property(replacement)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_descriptor_kind,
        patches[0].downstream_descriptor_kind,
        patches[0].installed_descriptor_kind,
    ) == ("property", "ordinary", "property")
    assert not findings


def test_v024_conditional_descriptor_variants_are_not_guessed(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    if feature_enabled():
        @property
        def state(self):
            return 1
    else:
        def state(self):
            return 2
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    @property
    def state(self):
        return 3
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = [relation for relation in relations if relation.relation == "override"]
    assert len(overrides) == 1
    assert (
        overrides[0].upstream_descriptor_kind,
        overrides[0].downstream_descriptor_kind,
        overrides[0].installed_descriptor_kind,
    ) == ("unknown", "property", "property")
    assert len(findings) == 1
    finding = findings[0]
    assert (
        finding.status,
        finding.reason_code,
        finding.generator_issue,
        finding.supplemental,
        finding.upstream_descriptor_kind,
        finding.downstream_descriptor_kind,
        finding.installed_descriptor_kind,
    ) == (
        "review",
        "conditional_descriptor_kind",
        False,
        True,
        "unknown",
        "property",
        "property",
    )


def test_v024_unknown_decorator_is_explicit_supplemental_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    @runtime_descriptor
    def run(self, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    overrides = [relation for relation in relations if relation.relation == "override"]
    assert len(overrides) == 1
    assert overrides[0].upstream_descriptor_kind == "unknown"
    assert len(findings) == 1
    assert (
        findings[0].reason_code,
        findings[0].supplemental,
        findings[0].upstream_descriptor_kind,
    ) == ("unknown_descriptor_kind", True, "unknown")


def test_v024_module_level_descriptor_wrappers_remain_values(
    tmp_path: Path,
) -> None:
    vllm_root, _ = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
def implementation(value):
    return value


as_property = property(implementation)
as_classmethod = classmethod(implementation)
as_staticmethod = staticmethod(implementation)
""",
    )

    index = generator.RepositoryIndex(vllm_root, "vllm")

    for name in ("as_property", "as_classmethod", "as_staticmethod"):
        qualified_name = f"vllm.base.{name}"
        assert {binding.kind for binding in index.find_final_bindings(qualified_name)} == {"value"}
        assert not index.find_final_callable_variants(qualified_name)


def test_v024_schema_v4_relations_load_with_unknown_descriptor_kinds(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "v4.jsonl"
    generator._write_jsonl(
        mapping,
        [
            {"_meta": {"schema": 4}},
            {
                "u": [
                    "vllm/base.py",
                    "Base",
                    "run",
                    ["sync", [], [["self", True]], None, [], None],
                ],
                "c": [
                    [
                        "override",
                        "vllm_ascend/plugin.py",
                        "Child",
                        "run",
                        ["sync", [], [["self", True]], None, [], None],
                    ]
                ],
                "e": [],
            },
        ],
    )

    relations = generator._load_compact_relations(mapping)

    assert len(relations) == 1
    assert (
        relations[0].upstream_descriptor_kind,
        relations[0].downstream_descriptor_kind,
        relations[0].installed_descriptor_kind,
    ) == (None, None, None)


def test_v024_schema_v5_round_trips_descriptor_kinds(
    tmp_path: Path,
) -> None:
    relation = generator.Relation(
        relation="monkey_patch",
        upstream_file="vllm/base.py",
        upstream_owner="Target",
        upstream_name="state",
        upstream_signature=["sync", [], [["self", True]], None, [], None],
        downstream_file="vllm_ascend/plugin.py",
        downstream_owner=None,
        downstream_name="replacement",
        downstream_signature=["sync", [], [["self", True]], None, [], None],
        evidence_file="vllm_ascend/plugin.py",
        evidence_line=8,
        upstream_descriptor_kind="property",
        downstream_descriptor_kind="ordinary",
        installed_descriptor_kind="property",
    )

    payloads = generator._relation_payloads(
        [relation],
        vllm_sha="upstream",
        ascend_sha="downstream",
    )
    assert payloads[0]["_meta"]["schema"] == 5
    assert payloads[1]["u"][-1] == "property"
    assert payloads[1]["c"][0][-2:] == ["ordinary", "property"]

    mapping = tmp_path / "v5.jsonl"
    generator._write_jsonl(mapping, payloads)
    loaded = generator._load_compact_relations(mapping)
    assert len(loaded) == 1
    assert (
        loaded[0].upstream_descriptor_kind,
        loaded[0].downstream_descriptor_kind,
        loaded[0].installed_descriptor_kind,
    ) == ("property", "ordinary", "property")


def test_v024_comparison_reports_descriptor_kind_changes_separately() -> None:
    common = {
        "relation": "override",
        "upstream_file": "vllm/base.py",
        "upstream_owner": "Base",
        "upstream_name": "state",
        "upstream_signature": ["sync", [], [["self", True]], None, [], None],
        "downstream_file": "vllm_ascend/plugin.py",
        "downstream_owner": "Child",
        "downstream_name": "state",
        "downstream_signature": ["sync", [], [["self", True]], None, [], None],
        "evidence_file": "vllm_ascend/plugin.py",
        "evidence_line": 5,
    }
    baseline = generator.Relation(
        upstream_descriptor_kind="ordinary",
        downstream_descriptor_kind="ordinary",
        installed_descriptor_kind="ordinary",
        **common,
    )
    generated = generator.Relation(
        upstream_descriptor_kind="property",
        downstream_descriptor_kind="ordinary",
        installed_descriptor_kind="ordinary",
        **common,
    )

    report = generator.compare_relations([generated], [baseline], [])

    assert report["summary"]["exact_matches"] == 1
    assert report["summary"]["descriptor_kind_changes"] == 1
    assert report["summary"]["old_only"] == 0
    assert report["summary"]["new_only"] == 0
    assert report["descriptor_kind_changes"] == [
        {
            "relation": generator._relation_label(generated),
            "baseline": ["ordinary", "ordinary", "ordinary"],
            "generated": ["property", "ordinary", "ordinary"],
        }
    ]


def test_v024_module_classmethod_replacement_keeps_installed_classmethod(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    @classmethod
    def run(cls, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


@classmethod
def replacement(cls, value):
    return value


Target.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_descriptor_kind,
        patches[0].downstream_descriptor_kind,
        patches[0].installed_descriptor_kind,
    ) == ("classmethod", "classmethod", "classmethod")
    assert not findings


def test_v024_typed_lazy_instance_patch_keeps_upstream_descriptor_for_review(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/platforms/interface.py",
        """
class Platform:
    @classmethod
    def verify(cls, value):
        return value
""",
    )
    _write(
        vllm_root,
        "vllm/platforms/__init__.py",
        """
from typing import TYPE_CHECKING
from .interface import Platform

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name):
    if name == "current_platform":
        return Platform()
    raise AttributeError(name)
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.platforms import current_platform


def replacement(value):
    return value


current_platform.verify = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_descriptor_kind,
        patches[0].downstream_descriptor_kind,
        patches[0].installed_descriptor_kind,
    ) == ("classmethod", "ordinary", None)
    assert len(findings) == 1
    assert (
        findings[0].reason_code,
        findings[0].supplemental,
        findings[0].upstream_descriptor_kind,
        findings[0].downstream_descriptor_kind,
        findings[0].installed_descriptor_kind,
    ) == (
        "descriptor_kind_mismatch",
        True,
        "classmethod",
        "ordinary",
        None,
    )


def test_v024_module_callable_patch_ignores_descriptor_decorator_uncertainty(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
@runtime_decorator
def run(value):
    return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
import vllm.base as base


def replacement(value):
    return value


base.run = replacement
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_descriptor_kind,
        patches[0].downstream_descriptor_kind,
        patches[0].installed_descriptor_kind,
    ) == (None, "ordinary", None)
    assert not [finding for finding in findings if finding.supplemental]


def test_v024_staticmethod_lambda_patch_records_wrapper_evidence(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Target:
    @staticmethod
    def run(value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Target


Target.run = staticmethod(lambda value: value)
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    patches = [relation for relation in relations if relation.relation == "monkey_patch"]
    assert len(patches) == 1
    assert (
        patches[0].upstream_descriptor_kind,
        patches[0].downstream_descriptor_kind,
        patches[0].installed_descriptor_kind,
    ) == ("staticmethod", "ordinary", "staticmethod")
    assert len(patches[0].evidence) == 1
    assert (
        patches[0].evidence[0].patch_kind,
        patches[0].evidence[0].installed_descriptor_kind,
    ) == ("staticmethod", "staticmethod")
    assert not findings


def test_v024_pinned_torch_inference_mode_is_an_ordinary_descriptor(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
import torch


class Base:
    @torch.inference_mode()
    def run(self, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    pinned_relations, pinned_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={
            "torch": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
        },
    ).generate()
    unregistered_relations, unregistered_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"torch": "unregistered-version"},
    ).generate()

    pinned = next(relation for relation in pinned_relations if relation.relation == "override")
    unregistered = next(relation for relation in unregistered_relations if relation.relation == "override")
    assert pinned.upstream_descriptor_kind == "ordinary"
    assert not pinned_findings
    assert unregistered.upstream_descriptor_kind == "unknown"
    assert len(unregistered_findings) == 1
    assert unregistered_findings[0].reason_code == "unknown_descriptor_kind"


def test_v024_outer_classmethod_determines_kind_after_unknown_inner_decorator(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    @classmethod
    @runtime_descriptor
    def run(cls, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    @classmethod
    def run(cls, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    override = next(relation for relation in relations if relation.relation == "override")
    assert (
        override.upstream_descriptor_kind,
        override.downstream_descriptor_kind,
        override.installed_descriptor_kind,
    ) == ("classmethod", "classmethod", "classmethod")
    assert not findings


def test_v024_outer_unknown_decorator_masks_inner_classmethod(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
class Base:
    @runtime_descriptor
    @classmethod
    def run(cls, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    @classmethod
    def run(cls, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    override = next(relation for relation in relations if relation.relation == "override")
    assert (
        override.upstream_descriptor_kind,
        override.downstream_descriptor_kind,
        override.installed_descriptor_kind,
    ) == ("unknown", "classmethod", "classmethod")
    assert len(findings) == 1
    assert findings[0].reason_code == "unknown_descriptor_kind"


def test_v024_shadowed_builtin_staticmethod_decorator_is_unknown(
    tmp_path: Path,
) -> None:
    vllm_root, ascend_root = _v018_source_roots(tmp_path)
    _write(
        vllm_root,
        "vllm/base.py",
        """
def runtime_descriptor(function):
    return function


staticmethod = runtime_descriptor


class Base:
    @staticmethod
    def run(self, value):
        return value
""",
    )
    _write(
        ascend_root,
        "vllm_ascend/plugin.py",
        """
from vllm.base import Base


class Child(Base):
    def run(self, value):
        return value
""",
    )

    relations, findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    override = next(relation for relation in relations if relation.relation == "override")
    assert (
        override.upstream_descriptor_kind,
        override.downstream_descriptor_kind,
        override.installed_descriptor_kind,
    ) == ("unknown", "ordinary", "ordinary")
    assert len(findings) == 1
    assert findings[0].reason_code == "unknown_descriptor_kind"
