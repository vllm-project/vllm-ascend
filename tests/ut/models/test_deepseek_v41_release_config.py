import pytest
from vllm import ModelRegistry
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config as UpstreamDeepseekV41Config

from vllm_ascend.config_utils import normalize_deepseek_v41_config
from vllm_ascend.models import register_model


def make_v41_config(**kwargs):
    return normalize_deepseek_v41_config(UpstreamDeepseekV41Config(**kwargs))


def _released_text_config():
    return {
        "model_type": "deepseek_v41_text",
        "num_hidden_layers": 40,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "engram_pad_token_id": 2,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }


def _rotation_config():
    return {
        "value_projection_rotated": True,
        "value_basis": "quarot_global",
        "key_and_gate_basis": "original",
        "runtime_delta_rotation": False,
    }


def test_released_config_names_are_available_to_existing_runtime():
    config = make_v41_config(
        architectures=["DeepseekV41ForCausalLM"],
        text_config=_released_text_config(),
        vision_config={
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 32,
            "max_image_tokens": 1024,
        },
        engram_rotation_config=_rotation_config(),
    )

    assert config.model_type == "deepseek_v41"
    assert config.kv_source_layers == config.kv_source_layer_ids
    assert config.index_source_layers == config.index_source_layer_ids
    assert config.candidate_source_layer == config.candidate_source_layer_id == 20
    assert config.engram_pad_id == config.engram_pad_token_id == 2
    assert config.dspark_n_activated_experts == 3
    assert config.dspark_num_experts_per_tok == 3
    assert config.vision_max_n_token == config.vision_max_n_token == 1024
    assert config.engram_rotation_config == _rotation_config()
    # The released checkpoint renamed the architecture to CausalLM but still
    # carries and serves the complete vision path.
    assert config.is_mm_prefix_lm
    assert config.mm_prefix_clamp_sliding_window
    assert config.mm_prefix_span_leading_pad_modulus == 2


def test_released_causal_architecture_uses_multimodal_wrapper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    register_model()

    assert any(
        args
        == (
            "DeepseekV41ForCausalLM",
            "vllm_ascend.models.deepseek_v41.vl_model:AscendDeepseekV41ForCausalLM",
        )
        for args, _kwargs in calls
    )


def test_legacy_conditional_config_remains_supported():
    config = make_v41_config(
        architectures=["DeepseekV41ForConditionalGeneration"],
        text_config={
            "model_type": "deepseek_v4.1_text",
            "kv_source_layers": [2],
            "index_source_layers": [2],
            "candidate_source_layer": 2,
            "engram_pad_id": 2,
            "dspark_n_activated_experts": 3,
        },
        vision_config={
            "model_type": "deepseek_v4.1_vision",
            "num_hidden_layers": 32,
            "max_image_tokens": 1024,
        },
    )

    assert config.kv_source_layer_ids == [2]
    assert config.engram_pad_token_id == 2
    assert config.dspark_num_experts_per_tok == 3
    assert config.is_mm_prefix_lm
    assert config.mm_prefix_span_leading_pad_modulus == 2


def test_release_and_legacy_aliases_cannot_disagree():
    text = _released_text_config()
    text["kv_source_layers"] = [3]
    with pytest.raises(ValueError, match="Conflicting DeepSeek V4.1 config fields"):
        make_v41_config(text_config=text)


def test_unknown_engram_rotation_contract_fails_closed():
    rotation = _rotation_config()
    rotation["runtime_delta_rotation"] = True
    with pytest.raises(ValueError, match="Unsupported DeepSeek V4.1 Engram rotation"):
        make_v41_config(
            text_config=_released_text_config(),
            engram_rotation_config=rotation,
        )


def test_ascend_registration_keeps_upstream_v41_frontend():
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_ascend.utils import adapt_patch

    adapt_patch(True)
    register_model()
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v41").__module__ == "vllm.tokenizers.deepseek_v41"
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v41").__module__ == "vllm.renderers.deepseek_v4"
