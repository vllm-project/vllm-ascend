# SPDX-License-Identifier: Apache-2.0
"""Source-level ownership and integration guards for Qwen4Exp on Ascend."""

from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.models.qwen4_exp.amd import indexer_qsa as upstream_indexer

from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa

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


def test_runner_uses_spec_decode_capabilities_not_qwen_type_checks() -> None:
    runner = _source("vllm_ascend/worker/model_runner_v1.py")
    # The concrete class is part of the static drafter union so mypy can see
    # the inherited proposer methods. Runtime dispatch still uses capabilities.
    assert "AscendQwen4ExpMTPProposer" in runner
    assert "isinstance(self.drafter, AscendQwen4ExpMTPProposer)" not in runner
    assert "SupportsPerGroupBlockTables" in runner
    assert "SupportsPerGroupKernelBlockSizes" in runner
    assert "SupportsAttentionBackendInitialization" in runner
    assert "SupportsCUDAGraphInitialization" in runner

    for proposer in (
        "vllm_ascend/spec_decode/qwen4_exp.py",
        "vllm_ascend/spec_decode/dspark_proposer.py",
        "vllm_ascend/spec_decode/gemma4_proposer.py",
    ):
        assert "def uses_per_group_kernel_block_sizes" in _source(proposer)


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


def test_qsa_normalization_uses_upstream_public_helper() -> None:
    qsa = _source("vllm_ascend/models/qwen4_exp/qsa.py")
    assert "upstream_indexer.apply_qsa_rmsnorm(" in qsa
    assert "upstream_indexer._gemma_rmsnorm(" not in qsa


def test_qsa_public_normalization_helper_matches_upstream(monkeypatch) -> None:
    indexer = object.__new__(ascend_qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_head_dim = 4
    indexer.k_layernorm = torch.nn.Identity()
    indexer.rotary_emb = SimpleNamespace(mrope_section=None)

    monkeypatch.setattr(
        ascend_qsa.envs,
        "VLLM_ASCEND_ENABLE_QSA_INDEXER_SPLIT_NORM_ROPE",
        False,
    )
    monkeypatch.setattr(
        ascend_qsa,
        "apply_qsa_rope",
        lambda _rotary_emb, _positions, tensor: tensor,
    )

    pooled = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    first_positions = torch.zeros(2, 3, dtype=torch.int64)
    actual = indexer.normalize_compressed_keys(pooled, first_positions)
    expected = upstream_indexer.apply_qsa_rmsnorm(
        indexer.k_layernorm,
        pooled.reshape(-1, indexer.index_head_dim),
    ).reshape(-1, 1, indexer.index_head_dim)

    torch.testing.assert_close(actual, expected)


def test_qsa_e3_custom_op_is_a3_scoped_and_has_meta_binding() -> None:
    build = _source("csrc/build_aclnn.sh")
    a3_start = build.index('elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]')
    a5_start = build.index('elif [[ "$SOC_VERSION" =~ ^ascend950')
    assert '"qsa_expand_e3"' in build[a3_start:a5_start]
    assert '"qsa_expand_e3"' not in build[a5_start:]

    binding = _source("csrc/torch_binding.cpp")
    meta = _source("csrc/torch_binding_meta.cpp")
    assert "qsa_expand_e3_out" in binding
    assert "qsa_expand_e3_out_meta" in meta
