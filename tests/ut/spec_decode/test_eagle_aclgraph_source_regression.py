# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the verified-main Eagle ACL graph patch."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ACLGRAPH = ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "autoregressive" / "aclgraph.py"
SPECULATOR = ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "autoregressive" / "speculator.py"
PATCH = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_v2" / "patch_eagle_speculator.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def test_eagle_aclgraph_uses_verified_main_speculator_contract() -> None:
    tree = ast.parse(ACLGRAPH.read_text())
    cls = _class(tree, "AutoRegressiveAclGraphManager")

    assert isinstance(cls.bases[0], ast.Name)
    assert cls.bases[0].id == "SpeculatorCudaGraphManager"

    source = ACLGRAPH.read_text()
    assert "AttentionStatePair" not in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source


def test_eagle_speculator_selects_backend_and_draft_replay_reuses_it() -> None:
    aclgraph_source = ACLGRAPH.read_text()
    speculator_source = SPECULATOR.read_text()

    assert "_get_graph_update_backend(self.attn_groups)" in speculator_source
    assert "issubclass(self.attn_backend" in speculator_source
    assert "set_current_vllm_config(self.vllm_config)" in aclgraph_source
    assert "attn_backend = self.speculator.attn_backend" in aclgraph_source
    assert "_get_graph_update_backend(self.speculator.attn_groups)" not in aclgraph_source


def test_eagle_patch_replaces_verified_main_manager_symbol() -> None:
    source = PATCH.read_text()

    assert "vllm_speculator_module.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager" in source
    assert "PrefillSpeculatorCudaGraphManager" not in source
    assert "DecodeSpeculatorCudaGraphManager" not in source
