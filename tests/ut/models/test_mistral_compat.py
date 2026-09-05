# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.mistral import (
    _get_mistral3_text_architectures,
    _get_mistral4_weights_mapper,
    _prepare_llama4_scaling,
    _prepare_mistral4_config,
    _prepare_mistral4_weights,
)


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_type": "mistral4",
        "rope_interleave": True,
        "q_lora_rank": 4,
        "kv_lora_rank": 2,
        "qk_head_dim": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "num_attention_heads": 2,
        "v_head_dim": 4,
        "hidden_size": 8,
        "moe_intermediate_size": 4,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 2.0,
            "partial_rotary_factor": 0.5,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_nested_mistral4_architecture() -> None:
    config = SimpleNamespace(model_type="mistral4")
    assert _get_mistral3_text_architectures(config) == [
        "Mistral4ForCausalLM"
    ]


def test_unknown_nested_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Mistral3 text_config"):
        _get_mistral3_text_architectures(SimpleNamespace(model_type="unknown"))


def test_mistral4_llama4_scaling_is_normalized() -> None:
    config = SimpleNamespace(
        rope_parameters={
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16384,
        }
    )
    _prepare_llama4_scaling(config)
    assert config.llama_4_scaling == {
        "beta": 0.1,
        "original_max_position_embeddings": 16384,
    }


def test_mistral4_config_is_normalized_for_mla() -> None:
    config = _config()
    _prepare_mistral4_config(config)
    assert "partial_rotary_factor" not in config.rope_parameters
    assert config.llama_4_scaling["beta"] == 0.1


def test_mistral4_explodes_packed_expert_weights() -> None:
    config = _config(n_routed_experts=2)
    gate_up = torch.arange(2 * 8 * 8).reshape(2, 8, 8)
    down = torch.arange(2 * 8 * 4).reshape(2, 8, 4)
    prepared = dict(
        _prepare_mistral4_weights(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.layers.0.mlp.experts.down_proj", down),
            ],
            config,
            channelwise_fp8=True,
        )
    )
    torch.testing.assert_close(
        prepared["model.layers.0.mlp.experts.1.gate_proj.weight"],
        gate_up[1, :4],
    )
    torch.testing.assert_close(
        prepared["model.layers.0.mlp.experts.0.down_proj.weight"], down[0]
    )


def test_mistral4_expands_channelwise_scales() -> None:
    config = _config()
    prepared = dict(
        _prepare_mistral4_weights(
            [
                (
                    "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
                    torch.tensor(4.0),
                ),
                (
                    "model.layers.0.self_attn.q_a_proj.activation_scale",
                    torch.tensor(5.0),
                ),
            ],
            config,
            channelwise_fp8=True,
        )
    )
    assert prepared[
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv"
    ].shape == (4, 1)
    assert not any(name.endswith("activation_scale") for name in prepared)


def test_mistral4_selects_mapper_from_constructed_attention() -> None:
    unfused = _get_mistral4_weights_mapper(
        ["model.layers.0.self_attn.q_a_proj.weight"]
    )
    fused = _get_mistral4_weights_mapper(
        ["model.layers.0.self_attn.fused_qkv_a_proj.weight"]
    )
    assert unfused._map_name(
        "model.layers.0.self_attn.q_a_proj.weight"
    ) == "model.layers.0.self_attn.q_a_proj.weight"
    assert fused._map_name(
        "model.layers.0.self_attn.q_a_proj.weight"
    ) == "model.layers.0.self_attn.fused_qkv_a_proj.weight"
