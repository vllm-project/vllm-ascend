# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the verified-main Eagle ACL graph patch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACLGRAPH = ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "eagle" / "aclgraph.py"
PATCH = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_v2" / "patch_eagle_speculator.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def test_eagle_aclgraph_uses_verified_main_speculator_contract() -> None:
    tree = ast.parse(ACLGRAPH.read_text())
    cls = _class(tree, "EagleAclGraphManager")

    assert isinstance(cls.bases[0], ast.Name)
    assert cls.bases[0].id == "SpeculatorCudaGraphManager"

    source = ACLGRAPH.read_text()
    assert "AttentionStatePair" not in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source


def test_eagle_patch_replaces_verified_main_manager_symbol() -> None:
    source = PATCH.read_text()

    assert "vllm_speculator_module.SpeculatorCudaGraphManager = EagleAclGraphManager" in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source
