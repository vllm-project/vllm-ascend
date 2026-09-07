# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for Ascend fused autoregressive draft decode."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPECULATOR = (
    ROOT
    / "vllm_ascend"
    / "worker"
    / "v2"
    / "spec_decode"
    / "autoregressive"
    / "speculator.py"
)
ACLGRAPH = SPECULATOR.with_name("aclgraph.py")
ATTN_UTILS = ROOT / "vllm_ascend" / "worker" / "v2" / "attn_utils.py"
GQA = ROOT / "vllm_ascend" / "attention" / "attention_v1.py"
MLA = ROOT / "vllm_ascend" / "attention" / "mla_v1.py"
DSA = ROOT / "vllm_ascend" / "attention" / "dsa_v1.py"
SFA = ROOT / "vllm_ascend" / "attention" / "sfa_v1.py"
DSA_CP = ROOT / "vllm_ascend" / "attention" / "context_parallel" / "dsa_cp.py"
INDEXER = ROOT / "vllm_ascend" / "attention" / "indexer.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found")


def test_speculator_reuses_upstream_fused_capture() -> None:
    source = SPECULATOR.read_text()
    cls = _class(ast.parse(source), "AscendAutoRegressiveSpeculator")

    method_names = {
        node.name for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    assert "_multi_step_decode" not in method_names
    assert "_fused_multi_step_decode" not in method_names
    assert "_generate_draft" not in method_names

    capture = ast.unparse(_method(cls, "capture"))
    assert "build_attn_metadata_wrapper()" in capture
    assert "disable_target_pcp_for_replicated_draft(self)" in capture
    assert "super().capture()" in capture
    assert "last_token_indices" not in capture
    assert "decode_cudagraph_manager.capture" not in capture


def test_aclgraph_manager_wraps_upstream_capture_only() -> None:
    source = ACLGRAPH.read_text()
    cls = _class(ast.parse(source), "AutoRegressiveAclGraphManager")
    capture = ast.unparse(_method(cls, "capture"))

    assert "super().capture" in capture
    assert "forward_fn.__name__" not in capture
    assert "_multi_step_decode" not in capture
    assert "create_forward_fn" not in capture
    assert "CudaGraphManager.capture" not in source
    assert "prepare_inputs_to_capture" not in source
    assert "draft_metadata_build_context" not in source
    assert "_captured_metadata_states" not in source


def test_all_executable_builders_enable_the_fused_update_hook() -> None:
    for path, class_name in (
        (GQA, "AscendAttentionMetadataBuilder"),
        (MLA, "AscendMLAMetadataBuilder"),
        (DSA, "AscendDSAMetadataBuilder"),
        (SFA, "AscendSFAMetadataBuilder"),
        (DSA_CP, "AscendDSACPMetadataBuilder"),
    ):
        cls = _class(ast.parse(path.read_text()), class_name)
        method = _method(cls, "update_draft_decode_metadata")
        source = ast.unparse(method)
        init = ast.unparse(_method(cls, "__init__"))

        assert "supports_draft_decode_metadata_update" in init
        assert "rebuild_draft_decode_metadata" not in source
        assert "vars(" not in source

    indexer = _class(ast.parse(INDEXER.read_text()), "AscendSFAIndexerMetadataBuilder")
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "supports_draft_decode_metadata_update"
            for target in node.targets
        )
        for node in indexer.body
    )
    _method(indexer, "update_draft_decode_metadata")


def test_fused_draft_uses_the_original_builder_entry_point() -> None:
    source = ATTN_UTILS.read_text()
    tree = ast.parse(source)
    build = ast.unparse(next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_attn_metadata"
    ))
    factory = ast.unparse(next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_draft_attn_metadata_factory"
    ))

    assert "_FUSED_DRAFT_METADATA_BUILD" not in source
    assert "_DRAFT_METADATA_RETAINED_STATES" not in source
    assert "attn_metadata_builder.build_for_drafting" not in build
    assert "attn_metadata_builder.build(" in build
    assert "DraftDecodeMetadataContext" not in build
    assert "draft_index" not in factory

    speculator = _class(ast.parse(SPECULATOR.read_text()), "AscendAutoRegressiveSpeculator")
    method = _method(speculator, "_build_draft_attn_metadata")
    factory_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_draft_attn_metadata_factory"
    )
    assert len(factory_call.args) == 3
