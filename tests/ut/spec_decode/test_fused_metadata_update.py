# SPDX-License-Identifier: Apache-2.0
"""Source regressions for Ascend fused draft metadata updates."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATTENTION = ROOT / "vllm_ascend" / "attention" / "attention_v1.py"
MLA = ROOT / "vllm_ascend" / "attention" / "mla_v1.py"
DSA = ROOT / "vllm_ascend" / "attention" / "dsa_v1.py"
SFA = ROOT / "vllm_ascend" / "attention" / "sfa_v1.py"
DSA_CP = ROOT / "vllm_ascend" / "attention" / "context_parallel" / "dsa_cp.py"
UTILS = ROOT / "vllm_ascend" / "attention" / "utils.py"
ATTN_UTILS = ROOT / "vllm_ascend" / "worker" / "v2" / "attn_utils.py"


def _method(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.unparse(method)


def _function(path: Path, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_generic_rebuild_shim_is_removed() -> None:
    sources = [
        UTILS.read_text(),
        ATTN_UTILS.read_text(),
        ATTENTION.read_text(),
        MLA.read_text(),
        DSA.read_text(),
        SFA.read_text(),
        DSA_CP.read_text(),
    ]
    assert all("rebuild_draft_decode_metadata" not in source for source in sources)
    assert all("register_draft_metadata_build" not in source for source in sources)


def test_builders_do_not_own_mutable_draft_state() -> None:
    sources = [
        ATTENTION.read_text(),
        MLA.read_text(),
        DSA.read_text(),
        SFA.read_text(),
        DSA_CP.read_text(),
    ]
    forbidden = (
        "_draft_decode_step",
        "_draft_common_attn_metadata",
        "_draft_build_kwargs",
        "_captured_draft_metadatas",
    )
    assert all(name not in source for source in sources for name in forbidden)


def test_sparse_metadata_classes_do_not_carry_rebuild_context() -> None:
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSAMetadata as AscendDSACPMetadata,
    )
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

    for metadata_cls in (
        AscendDSAMetadata,
        AscendDSACPMetadata,
        AscendSFAMetadata,
    ):
        assert "draft_context" not in metadata_cls.__dataclass_fields__


def test_sparse_rebuild_helpers_are_removed() -> None:
    source = UTILS.read_text()
    assert "DraftDecodeMetadataContext" not in source
    assert "replace_metadata_state_" not in source


def test_gqa_and_mla_update_host_tiling_metadata_in_place() -> None:
    gqa = _method(
        ATTENTION,
        "AscendAttentionMetadataBuilder",
        "update_draft_decode_metadata",
    )
    mla = _method(
        MLA,
        "AscendMLAMetadataBuilder",
        "update_draft_decode_metadata",
    )
    for source in (gqa, mla):
        assert "seq_lens_cpu.add_" in source
        assert "seq_lens_cpu.clamp_" in source
        assert ".tolist()" in source
        assert "build_for_drafting" not in source


def test_sparse_backends_update_live_device_metadata_in_place() -> None:
    dsa = _method(DSA, "AscendDSAMetadataBuilder", "update_draft_decode_metadata")
    sfa = _method(SFA, "AscendSFAMetadataBuilder", "update_draft_decode_metadata")
    dsa_cp = _method(DSA_CP, "AscendDSACPMetadataBuilder", "update_draft_decode_metadata")

    for source in (dsa, sfa, dsa_cp):
        assert "self.build(" not in source
        assert "build_for_drafting" not in source
        assert "draft_context" not in source
        assert "seq_lens_cpu" not in source

    assert "get_cos_and_sin_dsa" in dsa
    assert "raw_slot_mapping" in dsa
    assert "_build_sas_metadata" in dsa
    assert ".max()" not in dsa
    assert ".item()" not in dsa
    assert "get_cos_and_sin_mla" in sfa
    assert "get_cos_and_sin_dsa" in dsa_cp
    assert "_build_local_token_metadata" in dsa_cp
    assert "_build_sas_metadata" in dsa_cp
    assert ".max()" not in dsa_cp
    assert ".item()" not in dsa_cp


def test_draft_common_metadata_does_not_clone_or_advance_cpu_lengths() -> None:
    source = ast.unparse(_function(ATTN_UTILS, "build_attn_metadata"))
    assert "draft_seq_lens_cpu" not in source
    assert "seq_lens_cpu.add_" not in source
    assert "seq_lens_cpu_upper_bound" in source


def test_autoregressive_draft_build_does_not_use_draft_index() -> None:
    build = ast.unparse(_function(ATTN_UTILS, "build_attn_metadata"))
    factory = ast.unparse(
        next(
            node
            for node in ast.parse(ATTN_UTILS.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_draft_attn_metadata_factory"
        )
    )

    assert "build_for_drafting" not in build
    assert "draft_index" not in build
    assert "draft_index" not in factory
