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

_GENERATOR_PATH = (
    Path(__file__).parents[1]
    / "generate_interface_boundaries.py"
)
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
    imported_patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
        and relation.upstream_name == "external_hook"
    )
    assert imported_patch.downstream_file == (
        "vllm_ascend/implementation.py"
    )
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
    assert not any(
        relation.downstream_name == "local_only"
        for relation in relations
    )

    assert len(unresolved) == 6
    missing_target = next(
        relation
        for relation in unresolved
        if relation.relation == "monkey_patch"
        and relation.target_expression == "vllm.base.PatchTarget.missing"
    )
    assert missing_target.status == "risk"
    assert missing_target.reason_code == "possible_stale_patch"
    assert not missing_target.generator_issue
    injected = next(
        relation
        for relation in unresolved
        if relation.target_expression == "vllm.base.PatchTarget.injected"
    )
    assert injected.status == "expected"
    assert injected.reason_code == "inject_missing_member"
    inactive = next(
        relation
        for relation in unresolved
        if relation.target_expression == "vllm.base.PatchTarget.removed"
    )
    assert inactive.status == "excluded"
    assert inactive.reason_code == "inactive_guard"
    reachable_injection = next(
        relation
        for relation in unresolved
        if relation.target_expression == "vllm.base.PatchTarget.helper"
    )
    assert reachable_injection.status == "expected"
    assert reachable_injection.reason_code == "inject_missing_member"
    assert sum(
        relation.reason == "dynamic setattr attribute name"
        and relation.target_expression == "vllm.base.PatchTarget"
        and relation.status == "review"
        and relation.reason_code == "dynamic_setattr_name"
        and relation.generator_issue
        for relation in unresolved
    ) == 2
    assert not any(
        relation.target_expression == "vllm.base.PatchTarget.registry"
        for relation in unresolved
    )


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
    assert any(
        relation.relation == "override"
        and relation.downstream_name == "__init__"
        for relation in relations
    )


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

    patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    )
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
    assert [item.as_dict() for item in first_unresolved] == [
        item.as_dict()
        for item in second_unresolved
    ]
    assert sum("f" in payload for payload in first_payload) == 6


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

    assert index.canonical_name("vllm.wrapper.Base") == (
        "vllm.wrapper.Base"
    )
    assert index.find_class("vllm.wrapper.Base").file == (
        "vllm/wrapper.py"
    )
    assert index.find_class("vllm.wrapper.Base").resolved_bases == (
        "vllm.base.Base",
    )


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

    patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    )
    assert patch.upstream_file == "vllm/core.py"
    assert patch.upstream_name == "exported"
    assert not findings


def test_typed_lazy_export_resolves_to_its_interface_owner(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm-repo"
    ascend_root = tmp_path / "ascend-repo"
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/platforms/interface.py", """
class Platform:
    def verify(self, value):
        return value
""")
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

    patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    )
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert len(patches) == 2
    assert all(
        relation.downstream_name == "main_patch"
        for relation in patches
    )


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

    inheritance = next(
        relation
        for relation in relations
        if relation.relation == "inheritance"
    )
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
        relation.relation == "override"
        and relation.downstream_owner in {"Child", "OpaqueFirst"}
        for relation in relations
    )
    assert any(
        relation.relation == "override"
        and relation.reason.startswith("incomplete MRO")
        for relation in unresolved
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

    assert not [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert len(patches) == 2
    imported = next(
        relation
        for relation in patches
        if relation.downstream_name == "imported_patch"
    )
    assert imported.downstream_file == "vllm_ascend/implementation.py"
    assert len(imported.evidence) == 2
    assert {evidence.guards for evidence in imported.evidence} == {
        ("use_fast_path",),
        ("not (use_fast_path)",),
    }

    local = next(
        relation
        for relation in patches
        if relation.downstream_name == "local_patch"
    )
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/core.py"
    assert patches[0].upstream_name == "build"
    assert len(patches[0].evidence) == 2
    assert {
        evidence.target_expression
        for evidence in patches[0].evidence
    } == {"vllm.consumer.build"}
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert len(patches) == 1
    assert patches[0].upstream_file == "vllm/consumer.py"
    assert patches[0].upstream_owner == "Service"
    assert patches[0].upstream_name == "build"
    assert (
        patches[0].evidence[0].target_expression
        == "vllm.consumer.Service.build"
    )


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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert {relation.upstream_file for relation in patches} == {
        "vllm/third.py",
        "vllm/fourth.py",
    }
    assert all(relation.upstream_name == "load" for relation in patches)


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

    property_patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    )
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
    assert not any(
        relation.reason == "property patch is outside callable mapping scope"
        for relation in unresolved
    )


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
        if relation.downstream_owner == "Child"
        and relation.downstream_name == "hook"
    ]
    assert {relation.relation for relation in alias_relations} == {
        "monkey_patch",
        "override",
    }
    patch = next(
        relation
        for relation in alias_relations
        if relation.relation == "monkey_patch"
    )
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

    patches = {
        relation.upstream_name: relation
        for relation in relations
        if relation.relation == "monkey_patch"
    }
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
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

    patches = [
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    ]
    assert len(patches) == 1
    assert patches[0].upstream_name == "callable_target"
    assert len(findings) == 3
    assert sum(
        finding.status == "verified"
        and finding.reason_code == "field_mutation"
        for finding in findings
    ) == 2
    injected = next(
        finding
        for finding in findings
        if finding.reason_code == "inject_missing_field"
    )
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
    patch = next(
        relation
        for relation in relations
        if relation.relation == "monkey_patch"
    )
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
        relation.relation == "monkey_patch"
        and relation.upstream_owner == "Target"
        and relation.upstream_name == "hook"
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
        relation.relation == "monkey_patch"
        and relation.upstream_name == "hook"
        for relation in after_relations
    )
    risk = next(
        finding
        for finding in after_findings
        if finding.target_expression == "vllm.api.Target.hook"
    )
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
    risk = next(
        finding
        for finding in after_findings
        if finding.target_expression == "vllm.api.Target.new_hook"
    )
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
        relation.relation == "inheritance"
        and relation.upstream_name == "Base"
        and relation.downstream_owner == "Child"
        for relation in before_relations
    )
    assert not before_findings

    _write(vllm_root, "vllm/base.py", "")
    after_relations, after_findings = generator.InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
    ).generate()

    assert not any(
        relation.relation == "inheritance"
        and relation.downstream_owner == "Child"
        for relation in after_relations
    )
    risk = next(
        finding
        for finding in after_findings
        if finding.relation == "inheritance"
        and finding.downstream_owner == "Child"
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
        if relation.relation == "monkey_patch"
        and relation.upstream_name == "hook"
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
        if relation.relation == "monkey_patch"
        and relation.upstream_name == "hook"
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
