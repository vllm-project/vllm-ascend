# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DSpark compatibility tests; these do not claim PD inference coverage."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.model_loader.utils import get_model_cls
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig

from vllm_ascend.models import register_model
from vllm_ascend.models.glm5_dspark import Glm5DSparkForCausalLM, Glm5DSparkMLAAttention, Glm5DSparkModel
from vllm_ascend.patch.platform import patch_speculative_config
from vllm_ascend.patch.worker import patch_deepseek_v2


def test_registered_glm_draft_uses_glm_module(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    register_model()
    config = SimpleNamespace(
        model="/test/Glm5DSparkForCausalLM",
        convert_type="none",
        runner_type="generate",
        trust_remote_code=False,
        model_impl="vllm",
        hf_config=SimpleNamespace(architectures=["Glm5DSparkForCausalLM"]),
        registry=ModelRegistry,
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
    )
    assert get_model_cls(config) is Glm5DSparkForCausalLM
    assert Glm5DSparkForCausalLM.__module__ == "vllm_ascend.models.glm5_dspark"


@pytest.mark.parametrize("sliding_window_non_causal", [False, True])
def test_full_mla_draft_block_is_bidirectional(sliding_window_non_causal):
    # The training mask's full-attention case exposes all tokens within an
    # anchor block, independently of the sliding-window-only setting.
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[object()] * 5),
        config=SimpleNamespace(sliding_window_non_causal=sliding_window_non_causal),
    )
    assert Glm5DSparkForCausalLM.get_draft_attn_causal(model) == [False] * 5


def test_mla_cache_spec_matches_bidirectional_query_metadata(monkeypatch):
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        max_position_embeddings=128,
        num_hidden_layers=5,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 8000000},
    )
    attention = object()
    model = Glm5DSparkMLAAttention.__new__(Glm5DSparkMLAAttention)
    nn.Module.__init__(model)
    for name in ("DeepSeekV2FusedQkvAProjLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm"):
        monkeypatch.setattr(patch_deepseek_v2, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(patch_deepseek_v2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(patch_deepseek_v2, "get_rope", lambda *args, **kwargs: nn.Identity())
    # Exercise the real patched constructor. Mock only device-dependent
    # components so an unsupported keyword cannot be hidden by the test.
    with patch.object(
        patch_deepseek_v2, "MultiHeadLatentAttentionWrapper", return_value=SimpleNamespace(mla_attn=attention)
    ) as initialize:
        Glm5DSparkMLAAttention.__init__(
            model, vllm_config=SimpleNamespace(cache_config=object()), config=config, prefix="draft.layers.0.self_attn"
        )
    assert initialize.call_args.kwargs.get("non_causal_multi_token_decode", False) is True
    assert model.attn is attention


def test_aux_projection_preserves_order_not_mean_pooling():
    projection = nn.Linear(6, 2, bias=False)
    projection.input_size = 6
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]))
    model = SimpleNamespace(context_proj=projection, context_norm=nn.Identity())
    states = torch.arange(18, dtype=torch.float32).view(3, 6)
    actual = Glm5DSparkModel.combine_hidden_states(model, states)
    torch.testing.assert_close(actual, states @ projection.weight.T)
    with pytest.raises(ValueError, match="ordered auxiliary"):
        Glm5DSparkModel.combine_hidden_states(model, states[:, :2])


def context_model():
    layers = []
    context = torch.arange(12, dtype=torch.float32).view(3, 4)
    for index in range(2):
        output = context + index * 100
        attn = SimpleNamespace(
            q_lora_rank=2,
            fused_qkv_a_proj=MagicMock(return_value=(output, None)),
            rotary_emb=SimpleNamespace(cos_sin_cache=torch.arange(16).view(4, 4)),
            attn=SimpleNamespace(impl=SimpleNamespace(exec_kv_prefill=MagicMock()), kv_cache=object()),
        )
        layers.append(SimpleNamespace(self_attn=attn))
    return SimpleNamespace(layers=layers), context


@pytest.mark.parametrize("mapping_type", ["shared", "list", "tuple"])
def test_context_kv_uses_draft_rope_and_each_mla_projection(mapping_type):
    model, context = context_model()
    positions = torch.tensor([0, 2, 3])
    slots = torch.tensor([5, 6, 7])
    mapping = slots if mapping_type == "shared" else [slots, slots + 10]
    if mapping_type == "tuple":
        mapping = tuple(mapping)
    Glm5DSparkModel.precompute_and_store_context_kv(model, context, positions, mapping)
    for index, layer in enumerate(model.layers):
        attn = layer.self_attn
        args = attn.attn.impl.exec_kv_prefill.call_args.args
        torch.testing.assert_close(args[0], (context + index * 100)[:, 2:])
        cache = attn.rotary_emb.cos_sin_cache
        torch.testing.assert_close(args[1], cache[:, :2].repeat(1, 2)[positions, None, None])
        torch.testing.assert_close(args[2], cache[:, 2:].repeat(1, 2)[positions, None, None])
        assert args[3] is attn.attn.kv_cache
        assert args[4] is (slots if mapping_type == "shared" else mapping[index])
        attn.fused_qkv_a_proj.assert_called_once_with(context)


@pytest.mark.parametrize("mapping", [None, [None, None]])
def test_context_dummy_does_not_write_cache(mapping):
    model, context = context_model()
    Glm5DSparkModel.precompute_and_store_context_kv(model, context, torch.arange(3), mapping)
    for layer in model.layers:
        layer.self_attn.attn.impl.exec_kv_prefill.assert_not_called()


@pytest.mark.parametrize("case", ["layers", "positions", "slots"])
def test_context_rejects_incomplete_mapping(case):
    model, context = context_model()
    positions = torch.arange(2 if case == "positions" else 3)
    mapping = [torch.arange(2 if case == "slots" else 3)] * (1 if case == "layers" else 2)
    with pytest.raises(ValueError, match="must"):
        Glm5DSparkModel.precompute_and_store_context_kv(model, context, positions, mapping)


def test_weight_loader_keeps_mla_checkpoint_names():
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=4),
        target_vocab_size=4,
        draft_id_to_target_id=None,
        hf_to_vllm_mapper=Glm5DSparkForCausalLM.hf_to_vllm_mapper,
    )
    weights = [(name, torch.ones(2)) for name in ("t2d", "norm.weight", "context_proj.weight", "lm_head.weight")]
    loader = MagicMock()
    with patch("vllm_ascend.models.glm5_dspark.AutoWeightsLoader", return_value=loader):
        Glm5DSparkForCausalLM.load_weights(model, weights)
    names = [name for name, _ in loader.load_weights.call_args.args[0]]
    assert names == ["model.final_norm.weight", "model.context_proj.weight", "lm_head.weight"]
    assert model.has_own_lm_head and not model.has_own_embed_tokens
    assert not model.enable_confidence_head


@pytest.mark.parametrize("missing", ["d2t", "lm_head.weight"])
def test_reduced_vocab_must_have_mapping_and_head(missing):
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=4), target_vocab_size=8, draft_id_to_target_id=object())
    weights = [(name, torch.ones(2)) for name in ("d2t", "lm_head.weight") if name != missing]
    with pytest.raises(ValueError, match="Reduced-vocabulary"):
        Glm5DSparkForCausalLM.load_weights(model, weights)


def test_confidence_and_draft_mapping_inherit_upstream_contract():
    logits = torch.tensor([-3.0, 0.0, 3.0])
    model = SimpleNamespace(
        enable_confidence_head=True, model=SimpleNamespace(confidence_head=MagicMock(return_value=logits))
    )
    torch.testing.assert_close(
        Glm5DSparkForCausalLM.compute_confidence(model, torch.zeros(3, 4), torch.zeros(3, 2)), torch.sigmoid(logits)
    )
    model.draft_id_to_target_id = torch.tensor([0, 2, 4])
    ids = torch.arange(3)
    assert torch.equal(Glm5DSparkForCausalLM.map_draft_to_target(model, ids), torch.tensor([0, 3, 6]))


@pytest.mark.parametrize("architecture", ["Glm5DSparkForCausalLM", "Qwen3DSparkModel"])
def test_speculators_conversion_restores_only_glm_mla(monkeypatch, architecture):
    @dataclass
    class Arch:
        is_deepseek_mla: bool = False

    config = SpeculatorsConfig(
        architectures=["Qwen3DSparkModel"],
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    draft = SimpleNamespace(hf_config=config, model="/test/local-draft", model_arch_config=Arch())
    spec = SimpleNamespace(draft_model_config=draft, update_arch_=MagicMock())
    monkeypatch.setattr(SpeculatorsConfig, "get_config_dict", lambda *_: ({"architectures": [architecture]}, {}))
    patch_speculative_config._normalize_glm_mla_dspark(spec)
    is_mla = architecture == "Glm5DSparkForCausalLM"
    assert draft.hf_config.architectures == [architecture]
    assert draft.model_arch_config.is_deepseek_mla is is_mla
    assert spec.update_arch_.call_count == int(is_mla)


def test_glm_architecture_with_gqa_weights_is_rejected(monkeypatch):
    config = SpeculatorsConfig(architectures=["Qwen3DSparkModel"])
    spec = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=config, model="/test/gqa"))
    monkeypatch.setattr(
        SpeculatorsConfig, "get_config_dict", lambda *_: ({"architectures": ["Glm5DSparkForCausalLM"]}, {})
    )
    with pytest.raises(ValueError, match="cannot use GQA"):
        patch_speculative_config._normalize_glm_mla_dspark(spec)
