# SPDX-License-Identifier: Apache-2.0
"""Source-level ownership and MRV2 integration guards for Qwen4Exp."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_qwen4_exp_reuses_upstream_vllm_model_ownership() -> None:
    model = _source("vllm_ascend/models/qwen4_exp/model.py")
    qsa = _source("vllm_ascend/models/qwen4_exp/qsa.py")
    mtp = _source("vllm_ascend/models/qwen4_exp/mtp.py")

    assert "vllm.models.qwen4_exp.amd" in model
    assert "vllm.models.qwen4_exp.amd" in qsa
    assert "vllm.models.qwen4_exp.amd" in mtp
    assert "vllm.model_executor.models.qwen4_exp" not in model + qsa + mtp

    model_root = ROOT / "vllm_ascend/models/qwen4_exp"
    for platform_neutral_dir in ("amd", "common", "config", "nvidia"):
        assert not (model_root / platform_neutral_dir).exists()


def test_qwen4_exp_registry_uses_ascend_wrappers() -> None:
    registry = _source("vllm_ascend/models/__init__.py")
    for architecture, wrapper in (
        ("Qwen4ExpForCausalLM", "AscendQwen4ExpForCausalLM"),
        (
            "Qwen4ExpForConditionalGeneration",
            "AscendQwen4ExpForConditionalGeneration",
        ),
        ("Qwen4ExpMTP", "AscendQwen4ExpMTP"),
    ):
        assert f'"{architecture}"' in registry
        assert f"vllm_ascend.models.qwen4_exp:{wrapper}" in registry


def test_qwen4_exp_is_wired_only_through_mrv2() -> None:
    model = _source("vllm_ascend/models/qwen4_exp/model.py")
    model_state = _source("vllm_ascend/models/qwen4_exp/model_state.py")
    v2_attn = _source("vllm_ascend/worker/v2/attn_utils.py")
    v2_speculator = _source("vllm_ascend/worker/v2/spec_decode/target_dependent_ar/speculator.py")

    assert "get_model_state_cls" in model
    assert "AscendQwen4ExpModelState" in model
    assert "Qwen4ExpModelState" in model_state
    assert "AscendMambaHybridModelState" in model_state
    assert "build_qwen4_kv_cache_layout" in v2_attn
    assert "QSAMetadataBuilder" in v2_attn
    assert "AscendQSABackend" in v2_speculator
    assert 'self.attn_architecture = "QSA"' in v2_speculator


def test_qsa_hot_paths_do_not_add_host_tensor_syncs() -> None:
    qsa_sources = "\n".join(
        _source(path)
        for path in (
            "vllm_ascend/models/qwen4_exp/qsa.py",
            "vllm_ascend/models/qwen4_exp/ops.py",
            "vllm_ascend/models/qwen4_exp/lightning_indexer.py",
            "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
        )
    )
    assert ".item(" not in qsa_sources
    assert "torch.equal(" not in qsa_sources


def test_qsa_metadata_ignores_full_graph_dummy_request_boundaries() -> None:
    attn_utils = _source("vllm_ascend/worker/v2/attn_utils.py")
    branch_start = attn_utils.index("isinstance(attn_metadata_builder, QSAMetadataBuilder)")
    branch_end = attn_utils.index("metadata_variants =", branch_start)
    branch = attn_utils[branch_start:branch_end]

    assert "num_reqs > num_actual_reqs" in branch
    assert "query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu[" in branch
    assert ": num_actual_reqs + 1" in branch


def test_qsa_uses_fused_sparse_backends_and_upstream_qkv_projection() -> None:
    qsa = _source("vllm_ascend/models/qwen4_exp/qsa.py")
    assert "return qsa_select_paged_tokens_lightning(" in qsa
    assert "return qsa_gbsa_attention(" in qsa
    assert "torch.ops.vllm.triton_qsa_rmsnorm_mrope(" in qsa
    assert "torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(" not in qsa
    assert "def _project_qkv_gate(" not in qsa
    for fallback in (
        "qsa_select_paged_tokens_reference",
        "qsa_select_paged_tokens_triton",
        "qsa_sparse_paged_attention_reference",
        "qsa_sparse_paged_attention(",
        "VLLM_ASCEND_FORCE_QSA_REFERENCE",
        "VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER",
        "VLLM_ASCEND_ENABLE_QSA_INDEXER_FUSED_NORM_ROPE",
        "VLLM_ASCEND_ENABLE_QSA_MAIN_FUSED_NORM_ROPE",
    ):
        assert fallback not in qsa
