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


PatchTarget.hook = patched_hook
PatchTarget.missing = patched_hook
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

    assert len(unresolved) == 4
    missing_target = next(
        relation
        for relation in unresolved
        if relation.relation == "monkey_patch"
        and relation.target_expression == "vllm.base.PatchTarget.missing"
    )
    assert missing_target.status == "risk"
    assert missing_target.reason_code == "missing_upstream_member"
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
    assert any(
        relation.reason == "dynamic setattr attribute name"
        and relation.target_expression == "vllm.base.PatchTarget"
        and relation.status == "review"
        and relation.reason_code == "dynamic_setattr_name"
        and relation.generator_issue
        for relation in unresolved
    )
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
    assert sum("f" in payload for payload in first_payload) == 4


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
        and "opaque base before owned base" in relation.reason
        for relation in unresolved
    )
    assert any(
        relation.relation == "override"
        and relation.upstream_owner == "Partial"
        and relation.downstream_owner == "SafePrefix"
        and relation.downstream_name == "run"
        for relation in relations
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
