from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.e2e.vllm_interface.vllm_interface_contracts.cli import main as cli_main
from tests.e2e.vllm_interface.vllm_interface_contracts.generator import (
    InterfaceBoundaryGenerator,
    RepositoryIndex,
    _repository_index_from_file_fragments,
)
from tests.e2e.vllm_interface.vllm_interface_contracts.range_analysis import (
    GitSnapshot,
    analyze_range,
    discover_imports,
    validate_current_contracts,
    write_reports,
)


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Interface Test", "-c", "user.email=test@example.com", "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _repositories(tmp_path: Path, *, old_method: str, new_method: str) -> tuple[Path, Path, str, str, str]:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", f"class Base:\n    {old_method}\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/base.py", f"class Base:\n    {new_method}\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/child.py",
        "from vllm.base import Base\n\nclass Child(Base):\n    def run(self, value):\n        return value\n",
    )
    ascend_sha = _commit(ascend_root, "baseline")
    return vllm_root, ascend_root, old_sha, new_sha, ascend_sha


def _run(
    vllm_root: Path,
    ascend_root: Path,
    old_sha: str,
    new_sha: str,
    ascend_sha: str,
) -> dict[str, object]:
    return analyze_range(
        vllm_root=vllm_root,
        ascend_root=ascend_root,
        old=old_sha,
        new=new_sha,
        expect_ascend_sha=ascend_sha,
    )


def _call_repositories(
    tmp_path: Path,
    *,
    old_source: str,
    new_source: str,
    consumer_source: str,
) -> tuple[Path, Path, str, str, str]:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/api.py", old_source)
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/api.py", new_source)
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", consumer_source)
    ascend_sha = _commit(ascend_root, "baseline")
    return vllm_root, ascend_root, old_sha, new_sha, ascend_sha


def test_range_classifies_compatible_to_incompatible_override(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = _run(*roots)
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert any(item["relation"] == "override" for item in introduced)
    assert report["summary"]["introduced_break"] >= 1


def test_range_separates_preexisting_incompatibility(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    report = _run(*roots)
    overrides = [item for item in report["findings"] if item["relation"] == "override"]
    assert overrides
    assert overrides[0]["classification"] == "preexisting"
    assert overrides[0]["action"] == "review"
    assert overrides[0]["priority"] == "P2"
    assert overrides[0]["details"]["new_delta_on_preexisting_break"] is True
    assert [item["name"] for item in overrides[0]["details"]["parameter_delta"]["added"]] == ["new_required"]


def test_optional_only_override_is_review_without_exact_upstream_call(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=("class Base:\n    def __init__(self, value, optional=None):\n        self.value = value\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["action"] == "review"
    assert overrides[0]["priority"] == "P2"
    assert overrides[0]["details"]["new_optional_parameters"] == ["optional"]
    assert overrides[0]["details"]["upstream_call_evidence"] == []
    assert overrides[0]["suggestion"] == (
        "Review whether the new optional parameter can reach this downstream override at runtime. "
        "If it can, update the override signature and handle the new argument."
    )

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 0
    assert payload["summary"]["review_findings"] == 1
    assert payload["findings"] == []
    assert len(payload["review_findings"]) == 1
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "**Result: REVIEW**" in markdown
    assert (
        "The downstream override does not accept the new optional parameter `optional`, and the analyzer found no "
        "upstream call in this range that passes it to that override."
    ) in markdown
    assert "strict replacement contract" not in markdown


def test_optional_constructor_override_is_review_when_dispatch_is_not_proven(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self, value, optional=None):\n"
            "        self.value = value\n\n"
            "def build():\n"
            "    return Base(1, optional=True)\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "review"
    assert override["details"]["upstream_call_evidence"] == []
    candidates = override["details"]["candidate_upstream_call_evidence"]
    assert len(candidates) == 1
    assert candidates[0]["dispatch_kind"] == "direct_constructor"


def test_optional_override_stays_actionable_when_call_and_registration_prove_dispatch(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self, value, optional=None):\n"
            "        self.value = value\n\n"
            "def build():\n"
            "    return Base(1, optional=True)\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n\n"
            "def register():\n"
            "    from vllm.model_executor.custom_op import CustomOp\n"
            "    registry = {'Unrelated': Child}\n"
            "    registry.update({'Base': Child})\n"
            "    for name, op_cls in registry.items():\n"
            "        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["action"] == "modify"
    assert overrides[0]["priority"] == "P1"
    evidence = overrides[0]["details"]["upstream_call_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["matched_parameters"] == ["optional"]
    assert evidence[0]["file"] == "vllm/api.py"
    assert evidence[0]["dispatch_kind"] == "direct_constructor"
    assert evidence[0]["dispatch_proof"][0]["downstream_target"] == "vllm_ascend.consumer.Child"

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 1
    assert payload["summary"]["review_findings"] == 0
    assert len(payload["findings"]) == 1
    assert payload["review_findings"] == []


def test_optional_method_override_ignores_sibling_super_call(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source=(
            "class Base:\n"
            "    def run(self, value, optional=None): return value\n\n"
            "class Sibling(Base):\n"
            "    def invoke(self, value):\n"
            "        return super().run(value, optional=True)\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "review"
    assert override["details"]["upstream_call_evidence"] == []
    candidates = override["details"]["candidate_upstream_call_evidence"]
    assert len(candidates) == 1
    assert candidates[0]["dispatch_kind"] == "super_member"


def test_optional_method_override_is_actionable_for_defining_class_self_dispatch(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n    def run(self, value): return value\n    def invoke(self, value): return self.run(value)\n"
        ),
        new_source=(
            "class Base:\n"
            "    def run(self, value, optional=None): return value\n"
            "    def invoke(self, value): return self.run(value, optional=True)\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "modify"
    evidence = override["details"]["upstream_call_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["dispatch_kind"] == "self_member"
    assert evidence[0]["receiver_class"] == "vllm.api.Base"


def test_new_upstream_override_contract_conflict_is_introduced(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    pass\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = _run(*roots)
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["compatibility"]["old"]["exists"] is False
    assert overrides[0]["compatibility"]["new"]["exists"] is True


def test_signature_status_change_emits_fail_closed_finding(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source=(
            "def wrapper(fn): return fn\n\nclass Base:\n    @wrapper\n    def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = _run(*roots)
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "analysis_unresolved"
    assert overrides[0]["upstream"]["old"]["signature_status"] == "exact"
    assert overrides[0]["upstream"]["new"]["signature_status"] == "unknown"
    assert overrides[0]["change"] == "callable runtime signature contract changed"


def test_direct_import_relocation_is_an_introduced_break(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/old_location.py", "def helper(value): return value\n")
    old_sha = _commit(vllm_root, "old")
    _git(vllm_root, "mv", "vllm/old_location.py", "vllm/new_location.py")
    new_sha = _commit(vllm_root, "move")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm.old_location import helper\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    imports = [item for item in report["findings"] if item["relation"] == "direct_import"]
    assert len(imports) == 1
    assert imports[0]["classification"] == "introduced_break"
    assert "new_location.py" in imports[0]["change"]


def test_report_writer_separates_introduced_csv(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = _run(*roots)
    outputs = write_reports(report, tmp_path / "reports")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    introduced_csv = Path(outputs["introduced_csv"]).read_text(encoding="utf-8-sig")
    assert payload["schema_version"] == 10
    assert payload["metadata"]["vllm_old_sha"] == roots[2]
    assert "introduced_break" in introduced_csv
    assert "contract_kind" in introduced_csv.splitlines()[0]
    assert "preexisting" not in introduced_csv


def test_import_discovery_orders_module_and_symbol_references(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "vllm_ascend/imports.py",
        "import vllm\nfrom vllm import config\nvalue = vllm.config.ModelConfig\n",
    )
    references = discover_imports(tmp_path)
    assert [(item.module, item.symbol) for item in references] == [
        ("vllm", None),
        ("vllm", "config"),
        ("vllm.config.ModelConfig", None),
    ]


def test_import_detector_ignores_symbol_missing_at_both_endpoints(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/marker.py", "VALUE = 1\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm import never_existed\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    assert not [item for item in report["findings"] if item["relation"] == "direct_import"]


def test_import_detector_resolves_from_package_import_as_submodule(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/config.py", "VALUE = 1\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/config.py", "VALUE = 2\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm import config\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    assert not [item for item in report["findings"] if item["relation"] == "direct_import"]


def test_unchanged_verified_relationship_is_not_a_range_finding(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value): return value + 1",
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["source"] == "dynamic_relation_graph"]


def test_self_call_to_deleted_upstream_method_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["compatibility"]["old"]["compatible"] is True
    assert calls[0]["compatibility"]["new"]["exists"] is False
    assert calls[0]["details"]["lookup_root"] == "vllm.api.Base"
    assert calls[0]["details"]["resolution_basis"] == "old_fallback_self"


def test_deleted_self_call_through_stdlib_structural_base_is_detected(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("from abc import ABC\n\nclass Base(ABC):\n    def removed(self, value): return value\n"),
        new_source=("from abc import ABC\n\nclass Base(ABC):\n    pass\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["owner"] == "Base"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "missing"


def test_super_call_to_deleted_upstream_method_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def run(self, value):\n"
            "        return super().run(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["resolution_basis"] == "old_fallback_super"


def test_missing_dynamic_self_call_is_not_guessed_as_an_upstream_dependency(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    pass\n",
        new_source="class Base:\n    CHANGED = True\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.runtime_hook(value)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.runtime_hook"
    ]


def test_deleted_self_call_with_incomplete_mro_is_not_guessed(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from external_package import UnknownMixin\n"
            "from vllm.api import Base\n\n"
            "class Child(UnknownMixin, Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.removed"
    ]


def test_deleted_verified_override_target_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="pass",
    )
    report = _run(*roots)
    overrides = [item for item in report["findings"] if item["relation"] == "override"]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["compatibility"]["old"]["compatible"] is True
    assert overrides[0]["compatibility"]["new"]["exists"] is False
    assert overrides[0]["details"]["override_paths"] == [["vllm_ascend.child.Child.run", "vllm.base.Base.run"]]


def test_direct_call_required_parameter_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["direction"] == "downstream_call_to_upstream"
    assert calls[0]["details"]["call_shape"]["positional_count"] == 1


def test_triton_kernel_launch_required_parameter_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "from vllm.triton_utils import triton\n\n"
            "@triton.jit(do_not_specialize=['value'])\n"
            "def kernel(value, BLOCK_SIZE): pass\n"
        ),
        new_source=(
            "from vllm.triton_utils import triton\n\n"
            "@triton.jit(do_not_specialize=['value'])\n"
            "def kernel(value, required, BLOCK_SIZE): pass\n"
        ),
        consumer_source=("from vllm.api import kernel\n\ndef use():\n    kernel[(2,)](1, BLOCK_SIZE=16)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["compatibility"]["old"]["compatible"] is True
    assert calls[0]["compatibility"]["new"]["compatible"] is False
    assert calls[0]["details"]["invocation_kind"] == "triton_kernel_launch"
    assert calls[0]["evidence"][0]["callee"] == "kernel[2,]"


def test_direct_call_keyword_rename_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(*, value): return value\n",
        new_source="def helper(*, item): return item\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(value=1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"


def test_direct_constructor_call_uses_bound_init_signature(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Config:\n    def __init__(self, value): self.value = value\n",
        new_source="class Config:\n    def __init__(self, value, required): self.value = value\n",
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "direct"
    assert calls[0]["upstream"]["old"]["symbol_kind"] == "constructor"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "constructor"


def test_annotated_instance_method_call_is_checked(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Service:\n    def run(self, value): return value\n",
        new_source="class Service:\n    def run(self, value, required): return value\n",
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "instance"


def test_constructed_instance_method_call_is_checked(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    def __init__(self): pass\n    def run(self, value): return value\n"),
        new_source=("class Service:\n    def __init__(self): pass\n    def run(self, value, required): return value\n"),
        consumer_source=(
            "from vllm.api import Service\n\ndef use():\n    service = Service()\n    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call"
        and item["contract_kind"] == "call_arguments"
        and item["details"]["target"].endswith("Service.run")
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "instance"


def test_inherited_instance_method_is_resolved_in_each_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def run(self, value): return value\n\nclass Service(Base):\n    pass\n"),
        new_source=(
            "class Base:\n    def run(self, value, required): return value\n\nclass Service(Base):\n    pass\n"
        ),
        consumer_source=(
            "from vllm.api import Service\n\ndef use():\n    service = Service()\n    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call"
        and item["contract_kind"] == "call_arguments"
        and item["details"]["target"].endswith("Service.run")
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["owner"] == "Base"
    assert calls[0]["upstream"]["new"]["owner"] == "Base"


def test_inherited_constructor_is_resolved_in_each_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n    def __init__(self, value): self.value = value\n\nclass Config(Base):\n    pass\n"
        ),
        new_source=(
            "class Base:\n    def __init__(self, value, required): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["symbol_kind"] == "constructor"


def test_inherited_custom_new_keeps_constructor_analysis_unresolved(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n"
            "    def __new__(cls, *args): return super().__new__(cls)\n\n"
            "class Config(Base):\n"
            "    def __init__(self, value): self.value = value\n"
        ),
        new_source=(
            "class Base:\n"
            "    def __new__(cls, *args): return super().__new__(cls)\n\n"
            "class Config(Base):\n"
            "    def __init__(self, value, required): self.value = value\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["compatibility"]["old"]["exists"] is True
    assert calls[0]["compatibility"]["old"]["compatible"] is None


def test_inherited_metaclass_keeps_constructor_analysis_unresolved(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Meta(type):\n"
            "    def __call__(cls): return super().__call__()\n\n"
            "class Base(metaclass=Meta):\n"
            "    def __init__(self, value): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        new_source=(
            "class Meta(type):\n"
            "    def __call__(cls): return super().__call__()\n\n"
            "class Base(metaclass=Meta):\n"
            "    def __init__(self, value, required): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_shadowed_classmethod_decorator_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "def classmethod(fn): return fn\n\n"
            "class Service:\n"
            "    @classmethod\n"
            "    def build(cls, value): return cls()\n"
        ),
        new_source=(
            "def classmethod(fn): return fn\n\n"
            "class Service:\n"
            "    @classmethod\n"
            "    def build(cls, value, required): return cls()\n"
        ),
        consumer_source="from vllm.api import Service\n\ndef use():\n    return Service.build(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["upstream"]["old"]["descriptor"] == "unknown"


def test_overload_stubs_use_the_final_runtime_implementation(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "from typing import overload\n\n"
            "@overload\ndef helper(value: int) -> int: ...\n\n"
            "def helper(value): return value\n"
        ),
        new_source=(
            "from typing import overload\n\n"
            "@overload\ndef helper(value: int, required: int) -> int: ...\n\n"
            "def helper(value, required): return value\n"
        ),
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["signature_status"] == "exact"


def test_overload_only_target_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="from typing import overload\n\n@overload\ndef helper(value: int) -> int: ...\n",
        new_source=("from typing import overload\n\n@overload\ndef helper(value: int, required: int) -> int: ...\n"),
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_conditional_method_binding_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Service:\n    def run(self, value): return value\n",
        new_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value, required): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "unknown"


def test_changed_ambiguous_method_binding_emits_unresolved_delta(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        new_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value, required): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["compatibility"]["old"]["exists"] is None
    assert calls[0]["compatibility"]["new"]["exists"] is None


def test_classmethod_call_uses_bound_descriptor_signature(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    @classmethod\n    def build(cls, value): return cls()\n"),
        new_source=("class Service:\n    @classmethod\n    def build(cls, value, required): return cls()\n"),
        consumer_source=("from vllm.api import Service\n\ndef use():\n    return Service.build(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["descriptor"] == "classmethod"


def test_local_parameter_shadowing_does_not_create_a_direct_call(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source=("from vllm.api import helper\n\ndef use(helper):\n    return helper(1)\n"),
    )
    report = _run(*roots)
    assert report["summary"]["direct_call_dependencies"] == 0
    assert not [item for item in report["findings"] if item["relation"] == "direct_call"]


def test_constructed_instance_rebinding_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    def run(self, value): return value\n"),
        new_source=("class Service:\n    def run(self, value, required): return value\n"),
        consumer_source=(
            "from vllm.api import Service\n\n"
            "def use(other):\n"
            "    service = Service()\n"
            "    service = other\n"
            "    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item.get("details", {}).get("target", "").endswith("Service.run")
    ]


def test_dynamic_call_arguments_fail_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source=("from vllm.api import helper\n\ndef use(payload):\n    return helper(**payload)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_unknown_call_decorator_transform_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("def wrapper(fn): return fn\n\n@wrapper\ndef helper(value): return value\n"),
        new_source=("def wrapper(fn): return fn\n\n@wrapper\ndef helper(value, required): return value\n"),
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_direct_call_tuple_unpack_return_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return 1, 2\n",
        new_source="def helper(): return 1, 2, 3\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    left, right = helper()\n",
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"
    assert returns[0]["details"]["return_use"]["arity"] == 2


def test_direct_call_mapping_key_return_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return {'value': 1}\n",
        new_source="def helper(): return {'item': 1}\n",
        consumer_source=("from vllm.api import helper\n\ndef use():\n    return helper()['value']\n"),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_await_protocol_change_is_an_introduced_return_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="async def helper(): return 1\n",
        new_source="def helper(): return 1\n",
        consumer_source=("from vllm.api import helper\n\nasync def use():\n    return await helper()\n"),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_unused_direct_call_return_change_is_not_reported(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return 1, 2\n",
        new_source="def helper(): return 1, 2, 3\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper()\n",
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["contract_kind"] == "return_usage"]


def test_override_return_contract_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value, value\n",
        new_source="class Base:\n    def run(self, value): return value, value, value\n",
        consumer_source=(
            "from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value):\n        return value, value\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_transparent_super_return_follows_each_upstream_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def run(self) -> dict[str, int] | None:\n        return None\n"),
        new_source=(
            "class Base:\n"
            "    def run(self) -> tuple[dict[str, int] | None, dict[str, int] | None]:\n"
            "        return None, None\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def run(self) -> dict[str, int] | None:\n"
            "        return super().run()\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "compatibility_warning"
    assert returns[0]["compatibility"]["old"]["compatible"] is True
    assert returns[0]["compatibility"]["new"]["compatible"] is True
    downstream_return = returns[0]["details"]["downstream_return"]
    assert downstream_return["variants"] == [
        {
            "kind": "forward",
            "arity": None,
            "keys": (),
            "type_ref": "run",
        }
    ]
    assert "transparent_super_forward_precedes_annotation" in downstream_return["provenance"]


def test_classmethod_variadic_tuple_override_becomes_introduced_return_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[list[int], ...]:\n"
            "        return tuple([] for _ in range(2))\n"
        ),
        new_source=(
            "class Base:\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[tuple[list[int], ...], int]:\n"
            "        blocks = tuple([] for _ in range(2))\n"
            "        return blocks, 0\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[list[int], ...]:\n"
            "        return tuple([] for _ in range(2))\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"
    assert returns[0]["compatibility"]["old"]["compatible"] is True
    assert returns[0]["compatibility"]["new"]["compatible"] is False
    downstream_return = returns[0]["details"]["downstream_return"]
    assert downstream_return["status"] == "exact"
    assert downstream_return["variants"] == [
        {
            "kind": "tuple_variadic",
            "arity": None,
            "keys": (),
            "type_ref": None,
        }
    ]
    assert "unknown_return_transform" not in downstream_return["provenance"]


def test_super_call_is_a_direct_upstream_dependency(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Base:\n    def run(self, value): return value\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/base.py", "class Base:\n    def run(self, value, required): return value\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/child.py",
        (
            "from vllm.base import Base\n\n"
            "class Child(Base):\n"
            "    def run(self, value):\n"
            "        return super().run(value)\n"
        ),
    )
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    direct = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(direct) == 1
    assert direct[0]["classification"] == "introduced_break"


def test_current_pair_validation_reports_missing_call_target(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="OTHER = 1\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    dependencies, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert len(dependencies) == 1
    direct = [item for item in findings if item["relation"] == "direct_call"]
    assert len(direct) == 1
    assert direct[0]["status"] == "risk"
    assert "does not exist" in direct[0]["reason"]


def test_current_pair_validation_ignores_unverified_historical_self_candidate(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    dependencies, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert not [item for item in dependencies if item.callee == "self.removed"]
    assert not [
        item for item in findings if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.removed"
    ]


def test_validate_cli_reports_current_contract_counts(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    vllm_root, ascend_root, _, _, ascend_sha = roots
    output = tmp_path / "validation.json"
    assert (
        cli_main(
            [
                "validate",
                "--vllm-root",
                str(vllm_root),
                "--ascend-root",
                str(ascend_root),
                "--expect-ascend-sha",
                ascend_sha,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["direct_call_dependencies"] == 1
    assert payload["summary"]["contract_risks"] == 1
    assert payload["contract_findings"][0]["contract_kind"] == "call_arguments"


def test_vllm_interface_scenario_runs_only_upstream_pr_capabilities(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\ndef replacement(self, value):\n    return value\n\nBase.run = replacement\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    assert report["metadata"]["scenario"] == "vllm-interface"
    capabilities = report["metadata"]["analysis_plan"]["capabilities"]
    assert capabilities["inheritance_mro"]["state"] == "prerequisite"
    assert capabilities["monkey_patch"]["state"] == "skipped"
    assert capabilities["direct_import"]["state"] == "analyzed"
    assert report["metadata"]["timings_seconds"]["relation_generation.monkey_patch"] is None
    assert report["metadata"]["timings_seconds"]["direct_import_analysis"] is not None
    assert not any(item["relation"] == "monkey_patch" for item in report["findings"])


def test_vllm_interface_scenario_keeps_override_breaks(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert introduced
    assert {item["relation"] for item in introduced} == {"override"}


def test_generator_uses_lazy_mro_without_emitting_inheritance_relations(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source=("class Base:\n    def run(self, value): return value\n\nCURRENT_INTERFACE = True\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Parent(Base):\n"
            "    def run(self, value): return value\n\n"
            "class GrandChild(Parent):\n"
            "    def run(self, value): return value\n"
        ),
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )

    relations, findings = engine.generate()

    assert relations
    assert {relation.relation for relation in relations} == {"override"}
    assert {relation.downstream_owner for relation in relations} == {
        "Parent",
        "GrandChild",
    }
    assert not [finding for finding in findings if finding.relation == "inheritance"]
    assert engine.phase_timings["inheritance_mro"] is not None
    assert engine.phase_timings["override"] is not None


def test_vllm_interface_expands_transitive_override_impacts_under_one_root_cause(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Parent(Base):\n"
            "    def run(self, value): return value\n\n"
            "class GrandChild(Parent):\n"
            "    def run(self, value): return value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    introduced = [
        item
        for item in report["findings"]
        if item["classification"] == "introduced_break" and item["relation"] == "override"
    ]
    assert {item["downstream"]["owner"] for item in introduced} == {
        "Parent",
        "GrandChild",
    }
    by_owner = {item["downstream"]["owner"]: item for item in introduced}
    assert by_owner["Parent"]["details"]["impact_kind"] == "direct_override"
    assert by_owner["Parent"]["details"]["override_depth"] == 1
    assert by_owner["GrandChild"]["details"]["impact_kind"] == ("transitive_subclass_override")
    assert by_owner["GrandChild"]["details"]["override_depth"] == 2
    assert by_owner["GrandChild"]["details"]["override_paths"] == [
        [
            "vllm_ascend.consumer.GrandChild.run",
            "vllm_ascend.consumer.Parent.run",
            "vllm.api.Base.run",
        ]
    ]

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 10
    assert payload["summary"]["introduced_breaks"] == 2
    assert payload["summary"]["root_causes"] == 1
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert (
        "Override path: `vllm_ascend.consumer.GrandChild.run -> vllm_ascend.consumer.Parent.run -> vllm.api.Base.run`"
    ) in markdown


def test_vllm_interface_reports_import_and_call_breaks_as_one_root_cause(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="OTHER = 1\n",
        consumer_source=("from vllm.api import helper\n\ndef use():\n    return helper(1)\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert {item["relation"] for item in introduced} == {
        "direct_call",
        "direct_import",
    }
    output_dir = tmp_path / "upstream-report"
    outputs = write_reports(report, output_dir)
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 2
    assert payload["summary"]["root_causes"] == 1
    assert {item["relation"] for item in payload["findings"]} == {
        "direct_call",
        "direct_import",
    }
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "downstream imports, overrides" in markdown
    assert "direct imports" not in markdown
    assert markdown.count("- Upstream: `vllm/api.py:helper`") == 2


def test_vllm_interface_reports_masked_preexisting_delta_as_review(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
    )
    assert report["summary"]["preexisting"] >= 1
    outputs = write_reports(report, tmp_path / "upstream-reports")
    assert Path(outputs["markdown"]).name == "vllm-interface-pr-summary.md"
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 0
    assert payload["summary"]["review_findings"] == 1
    assert payload["findings"] == []
    assert len(payload["review_findings"]) == 1
    assert payload["review_findings"][0]["details"]["new_delta_on_preexisting_break"] is True
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "historical incompatibilities are intentionally excluded" in markdown
    assert (
        "This range adds another contract difference, but the downstream code was already incompatible at the "
        "old revision."
    ) in markdown


def test_vllm_interface_cli_log_keeps_only_exact_masked_delta_review(
    tmp_path: Path,
    capsys,
) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    output_dir = tmp_path / "ci-output"
    assert (
        cli_main(
            [
                "analyze-range",
                "--vllm-root",
                str(roots[0]),
                "--ascend-root",
                str(roots[1]),
                "--old",
                roots[2],
                "--new",
                roots[3],
                "--expect-ascend-sha",
                roots[4],
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    console = capsys.readouterr().out
    assert '"introduced_breaks": 0' in console
    assert '"review_findings": 1' in console
    assert '"preexisting":' not in console
    assert "new_delta_masked_by_preexisting_incompatibility" in console
    assert "analysis_unresolved" not in console


def test_parallel_analysis_matches_serial_results(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def run(self, value): return value\n\ndef helper(value): return value\n"),
        new_source=("class Base:\n    def run(self, value, required): return value\n\nOTHER = 1\n"),
        consumer_source=(
            "from vllm.api import Base, helper\n\n"
            "class Child(Base):\n"
            "    def run(self, value): return value\n\n"
            "def use():\n"
            "    return helper(1)\n"
        ),
    )
    serial = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        analysis_workers=1,
    )
    parallel = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        analysis_workers=3,
    )

    assert parallel["findings"] == serial["findings"]
    assert parallel["summary"] == serial["summary"]
    assert serial["metadata"]["execution"]["parallel_branches"] is False
    assert parallel["metadata"]["execution"]["parallel_branches"] is True
    assert parallel["metadata"]["execution"]["analysis_workers_used"] == 3


def test_serial_file_fragments_match_repository_index(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _write(root, "vllm/__init__.py", "")
    _write(root, "vllm/accessors.py", "def read(self): return self._value\n")
    _write(
        root,
        "vllm/model.py",
        (
            "from vllm.accessors import read\n\n"
            "class Model:\n"
            "    value = property(read)\n\n"
            "    def run(self, item): return item\n"
        ),
    )

    regular = RepositoryIndex(root, "vllm")
    fragmented = RepositoryIndex._from_serial_file_fragments(root, "vllm")

    def snapshot(index: RepositoryIndex) -> dict[str, object]:
        return {
            "modules": sorted(index.modules),
            "classes": {
                name: (item.bases, item.resolved_bases, sorted(item.methods))
                for name, item in sorted(index.classes.items())
            },
            "callables": {
                name: (
                    item.signature,
                    item.descriptor_kind,
                    item.decorator_references,
                    item.property_accessors,
                )
                for name, item in sorted(index.callables.items())
            },
            "aliases": dict(sorted(index.aliases.items())),
            "exports": sorted(index.unconditional_exports),
            "symbols": sorted(index.unconditional_symbols),
        }

    assert snapshot(fragmented) == snapshot(regular)


def test_process_file_fragments_reuse_unchanged_git_blobs(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _write(root, "vllm/__init__.py", "")
    _write(root, "vllm/accessors.py", "def read(self): return self._value\n")
    _write(
        root,
        "vllm/model.py",
        "class Model:\n    def run(self, item): return item\n",
    )
    first_sha = _commit(root, "first")
    cache_dir = tmp_path / "file-cache"

    cold, cold_status = _repository_index_from_file_fragments(
        root,
        "vllm",
        ordinary_descriptor_decorators=frozenset(),
        source_version=first_sha,
        cache_dir=cache_dir,
        index_workers=2,
    )
    hot, hot_status = _repository_index_from_file_fragments(
        root,
        "vllm",
        ordinary_descriptor_decorators=frozenset(),
        source_version=first_sha,
        cache_dir=cache_dir,
        index_workers=2,
    )
    assert cold_status["status"] == "miss"
    assert cold_status["workers_used"] == 2
    assert cold_status["hit_ratio"] == 0.0
    assert cold_status["database_bytes"] > 0
    assert hot_status["status"] == "hit"
    assert hot_status["cache_hits"] == 3
    assert hot_status["hit_ratio"] == 1.0
    assert sorted(hot.callables) == sorted(cold.callables)

    _write(
        root,
        "vllm/model.py",
        "class Model:\n    def run(self, item, optional=None): return item\n",
    )
    second_sha = _commit(root, "second")
    incremental, incremental_status = _repository_index_from_file_fragments(
        root,
        "vllm",
        ordinary_descriptor_decorators=frozenset(),
        source_version=second_sha,
        cache_dir=cache_dir,
        index_workers=2,
    )
    regular = RepositoryIndex(root, "vllm")

    assert incremental_status["status"] == "partial_hit"
    assert incremental_status["cache_hits"] == 2
    assert incremental_status["cache_misses"] == 1
    assert incremental_status["hit_ratio"] == 0.666667
    assert (
        incremental.callables["vllm.model.Model.run"].signature == regular.callables["vllm.model.Model.run"].signature
    )


def test_downstream_repository_index_cache_miss_then_hit(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    @classmethod\n    def run(cls, value): return value\n"),
        new_source=("class Base:\n    @classmethod\n    def run(cls, value, required): return value\n"),
        consumer_source=(
            "from vllm.api import Base\n\nclass Child(Base):\n    @classmethod\n    def run(cls, value): return value\n"
        ),
    )
    cache_dir = tmp_path / "repository-index-cache"
    first = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        analysis_workers=1,
        downstream_index_cache_dir=cache_dir,
    )
    second = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        analysis_workers=1,
        downstream_index_cache_dir=cache_dir,
    )

    first_cache = first["metadata"]["repository_index_cache"]["downstream"]
    second_cache = second["metadata"]["repository_index_cache"]["downstream"]
    assert first_cache["status"] == "miss"
    assert second_cache["status"] == "hit"
    assert first_cache["key"] == second_cache["key"]
    assert len(list(cache_dir.glob("vllm_ascend-*.pickle"))) == 1
    assert second["findings"] == first["findings"]
    assert second["summary"] == first["summary"]

    Path(str(second_cache["path"])).write_bytes(b"invalid cache")
    rebuilt = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        analysis_workers=1,
        downstream_index_cache_dir=cache_dir,
    )
    rebuilt_cache = rebuilt["metadata"]["repository_index_cache"]["downstream"]
    assert rebuilt_cache["status"] == "invalid_rebuilt"
    assert rebuilt_cache["reason"].startswith("UnpicklingError:")
    assert rebuilt["findings"] == first["findings"]
    assert rebuilt["summary"] == first["summary"]
