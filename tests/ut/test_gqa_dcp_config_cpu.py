# SPDX-License-Identifier: Apache-2.0
"""Check replicated GQA configuration without importing the NPU runtime."""

import ast
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_validation():
    root = Path(__file__).resolve().parents[2] / "vllm_ascend"
    functions = []
    for filename, name in (
        ("utils.py", "is_kimi_k3_gqa_dspark"),
        ("platform.py", "_validate_draft_decode_context_parallel_config"),
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        functions.append(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))
    namespace = {}
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=functions, type_ignores=[]))
    exec(compile(source, "gqa_dcp_config", "exec"), namespace)
    return namespace["_validate_draft_decode_context_parallel_config"]


def _config(dcp_size=2):
    target = SimpleNamespace(
        architectures=["KimiK3ForConditionalGeneration"],
        hf_config=SimpleNamespace(model_type="kimi_k3"),
    )
    draft = SimpleNamespace(
        use_mla=False,
        architectures=["Qwen3DSparkModel"],
        hf_config=SimpleNamespace(model_type="qwen3"),
        model_arch_config=SimpleNamespace(total_num_attention_heads=64),
        get_total_num_kv_heads=lambda: 16,
    )
    return SimpleNamespace(
        model_config=target,
        parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=dcp_size),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens_per_batch_size=None,
            target_model_config=target,
            draft_model_config=draft,
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=1),
        ),
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_replicated_gqa_keeps_target_dcp_and_draft_tp8(dcp_size):
    config = _config(dcp_size)
    _load_validation()(config)
    assert config.parallel_config.decode_context_parallel_size == dcp_size
    assert config.parallel_config.tensor_parallel_size == 8
    assert config.speculative_config.draft_parallel_config.decode_context_parallel_size == 1
    assert config.speculative_config.draft_parallel_config.tensor_parallel_size == 8


def test_replicated_gqa_rejects_a_different_draft_tp():
    config = _config()
    config.speculative_config.draft_parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="draft tensor parallel size"):
        _load_validation()(config)


def test_replicated_gqa_requires_target_tp_divisible_by_dcp():
    with pytest.raises(ValueError, match="must be divisible"):
        _load_validation()(_config(3))


@pytest.mark.parametrize("different", ["target", "draft", "method"])
def test_other_models_keep_gqa_dcp_head_constraints(different):
    config = _config()
    if different == "target":
        config.model_config.architectures = ["Qwen3ForCausalLM"]
        config.model_config.hf_config.model_type = "qwen3"
    elif different == "draft":
        config.speculative_config.draft_model_config.architectures = ["Qwen3ForCausalLM"]
    else:
        config.speculative_config.method = "eagle"
    with pytest.raises(ValueError, match="must be greater than total num kv heads"):
        _load_validation()(config)


def test_mla_draft_keeps_existing_dcp_exemption():
    config = _config()
    config.speculative_config.draft_model_config.architectures = ["K3DSparkModel"]
    config.speculative_config.draft_model_config.hf_config.model_type = "k3_dspark"
    config.speculative_config.draft_model_config.use_mla = True
    _load_validation()(config)


def test_dynamic_dspark_keeps_existing_dcp_rejection():
    config = _config()
    config.speculative_config.num_speculative_tokens_per_batch_size = {1: 3}
    with pytest.raises(ValueError, match="Dynamic speculative decoding"):
        _load_validation()(config)


def test_replicated_draft_config_preserves_target_model_and_isolates_cache():
    @dataclass
    class Config:
        model_config: object
        parallel_config: object
        cache_config: object
        kv_transfer_config: object

        def __post_init__(self):
            # Ascend Flash validation inspects the target model even when
            # this config is used to construct the separate GQA drafter.
            assert self.model_config.use_mla

    class Upstream:
        def _create_draft_vllm_config(self):
            return self.vllm_config

    root = Path(__file__).resolve().parents[2] / "vllm_ascend/spec_decode"
    namespace = {"copy": copy, "replace": replace, "SpecDecodeBaseProposer": Upstream}
    for filename, class_name in (
        ("llm_base_proposer.py", "AscendSpecDecodeBaseProposer"),
        ("dspark_proposer.py", "AscendDSparkProposer"),
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        cls.body = [
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_create_draft_vllm_config"
        ]
        exec(compile("from __future__ import annotations\n" + ast.unparse(cls), filename, "exec"), namespace)
        namespace["AscendDflashProposer"] = namespace["AscendSpecDecodeBaseProposer"]

    target_model = SimpleNamespace(use_mla=True, runner_type="generate", hf_config=object())
    target_parallel = SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=2, rank=5)
    target = Config(target_model, target_parallel, SimpleNamespace(block_size=768), object())
    proposer = namespace["AscendDSparkProposer"]()
    proposer.vllm_config = target
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(use_mla=False, runner_type="draft"),
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=2, rank=0),
    )
    proposer._uses_dcp_replicated_draft_kv = lambda: True
    draft = proposer._create_draft_vllm_config()

    assert draft.model_config.use_mla
    assert draft.model_config.hf_config is target_model.hf_config
    assert draft.model_config.runner_type == "draft" and target_model.runner_type == "generate"
    assert draft.parallel_config.tensor_parallel_size == 8
    assert draft.parallel_config.decode_context_parallel_size == 1
    assert draft.parallel_config.rank == 5
    assert target_parallel.decode_context_parallel_size == 2
    assert draft.cache_config is not target.cache_config
    draft.cache_config.block_size = 128
    assert target.cache_config.block_size == 768
    assert draft.kv_transfer_config is None and target.kv_transfer_config is not None
