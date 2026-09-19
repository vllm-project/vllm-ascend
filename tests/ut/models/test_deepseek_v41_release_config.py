import json

import pytest

pytest.importorskip("vllm.transformers_utils.configs.deepseek_v41", reason="V4.1 is unavailable on vLLM 0.28")

from vllm import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config as UpstreamDeepseekV41Config

from vllm_ascend.models import register_model
from vllm_ascend.utils import normalize_deepseek_v41_config


def make_v41_config(**kwargs):
    return normalize_deepseek_v41_config(UpstreamDeepseekV41Config(**kwargs))


def test_released_config_loads_through_vllm_registry(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": _released_text_config(),
                "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, UpstreamDeepseekV41Config)
    assert config.is_mm_prefix_lm
    assert config.mm_prefix_span_leading_pad_modulus == 2


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


def test_released_config_names_are_available_to_runtime():
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
    for name, value in _released_text_config().items():
        if name != "model_type":
            assert getattr(config, name) == value
    assert config.vision_max_n_token == 1024
    # Runtime defaults must not manufacture the private port's old aliases.
    for name in (
        "kv_source_layers",
        "index_source_layers",
        "candidate_source_layer",
        "engram_pad_id",
        "dspark_n_activated_experts",
    ):
        assert not hasattr(config, name)
    assert config.engram_rotation_config == _rotation_config()
    # The released CausalLM architecture still carries the complete vision path.
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


def test_engram_rotation_contract_is_preserved():
    rotation = _rotation_config()
    rotation["runtime_delta_rotation"] = True
    config = make_v41_config(
        text_config=_released_text_config(),
        engram_rotation_config=rotation,
    )
    assert config.engram_rotation_config == rotation


def test_ascend_registration_keeps_upstream_v41_frontend():
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    from vllm_ascend.utils import adapt_patch

    adapt_patch(True)
    register_model()
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v41").__module__ == "vllm.tokenizers.deepseek_v41"
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v41").__module__ == "vllm.renderers.deepseek_v4"
