# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers import rotary_embedding

from vllm_ascend.patch.platform.patch_speculative_config import _normalize_kimi_dflash_rope


def _kimi_dflash_config_dict():
    # Metadata of z-lab/Kimi-K2.5-DFlash, also used by the Kimi-K2.6 nightly.
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 7168,
        "vocab_size": 163840,
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 6,
        "num_target_layers": 61,
        "max_position_embeddings": 262144,
        "rope_theta": 50000.0,
        "dflash_config": {"target_layer_ids": [1, 12, 24, 35, 47, 58], "mask_token_id": 163838},
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 64.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    }


def _config_with_rope_field(field):
    values = _kimi_dflash_config_dict()
    values[field] = values.pop("rope_scaling")
    return SimpleNamespace(**values)


def _rope_parameters(config):
    parameters = getattr(config, "rope_parameters", None)
    return parameters if parameters is not None else config.rope_scaling


def test_kimi_dflash_hf_override_restores_legacy_yarn_amplitude():
    config = Qwen3Config(**_kimi_dflash_config_dict())
    original_rope = _rope_parameters(config)
    original_values = deepcopy(original_rope)

    normalized = SpeculativeConfig.hf_config_override(config)

    assert normalized is config
    assert normalized.architectures == ["DFlashDraftModel"]
    assert _rope_parameters(normalized)["attention_factor"] == pytest.approx(1.4158883083359672)
    # The checkpoint's input dictionary may also be held by a target config.
    assert original_rope == original_values
    assert {
        key: value for key, value in _rope_parameters(normalized).items() if key != "attention_factor"
    } == original_values


@pytest.mark.parametrize("field", ["rope_parameters", "rope_scaling"])
def test_kimi_dflash_accepts_both_config_representations(field):
    config = _config_with_rope_field(field)
    original_rope = getattr(config, field)
    original_values = deepcopy(original_rope)

    _normalize_kimi_dflash_rope(config)
    first_result = deepcopy(vars(config))
    _normalize_kimi_dflash_rope(config)

    assert getattr(config, field)["attention_factor"] == pytest.approx(1.4158883083359672)
    assert original_rope == original_values
    assert vars(config) == first_result


@pytest.mark.parametrize("attention_factor", [0.0, 1.0, 1.7])
@pytest.mark.parametrize("field", ["rope_parameters", "rope_scaling"])
def test_kimi_dflash_preserves_explicit_attention_factor(field, attention_factor):
    config = _config_with_rope_field(field)
    getattr(config, field)["attention_factor"] = attention_factor
    before = deepcopy(vars(config))

    _normalize_kimi_dflash_rope(config)

    assert vars(config) == before


def test_kimi_dflash_none_attention_factor_uses_legacy_default():
    config = _config_with_rope_field("rope_parameters")
    config.rope_parameters["attention_factor"] = None

    _normalize_kimi_dflash_rope(config)

    assert config.rope_parameters["attention_factor"] == pytest.approx(1.4158883083359672)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "llama"),
        ("architectures", ["Qwen3ForCausalLM"]),
        ("architectures", ["DFlash2DraftModel"]),
        ("hidden_size", 4096),
        ("vocab_size", 151936),
        ("num_target_layers", 32),
        ("dflash_config", {"target_layer_ids": [1, 6, 12, 18, 24, 29]}),
    ],
)
def test_other_draft_configs_are_not_rewritten(field, value):
    config = _config_with_rope_field("rope_parameters")
    setattr(config, field, value)
    before = deepcopy(vars(config))

    _normalize_kimi_dflash_rope(config)

    assert vars(config) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rope_type", "deepseek_yarn"),
        ("factor", 32.0),
        ("mscale", 0.5),
        ("mscale_all_dim", 0.5),
        ("original_max_position_embeddings", 8192),
    ],
)
def test_other_yarn_profiles_are_not_rewritten(field, value):
    config = _config_with_rope_field("rope_parameters")
    config.rope_parameters[field] = value
    before = deepcopy(vars(config))

    _normalize_kimi_dflash_rope(config)

    assert vars(config) == before


def test_kimi_dflash_without_rope_config_is_unchanged():
    values = _kimi_dflash_config_dict()
    del values["rope_scaling"]
    config = SimpleNamespace(**values)
    before = deepcopy(vars(config))

    _normalize_kimi_dflash_rope(config)

    assert vars(config) == before


def test_kimi_dflash_prefers_canonical_rope_parameters():
    config = _config_with_rope_field("rope_parameters")
    config.rope_parameters["attention_factor"] = 1.7
    config.rope_scaling = deepcopy(_kimi_dflash_config_dict()["rope_scaling"])
    before = deepcopy(vars(config))

    _normalize_kimi_dflash_rope(config)

    assert vars(config) == before


def test_kimi_dflash_get_rope_cache_matches_legacy_yarn(monkeypatch):
    config = Qwen3Config(**_kimi_dflash_config_dict())
    reference_parameters = dict(_rope_parameters(config))
    # vLLM 0.29 did not forward these two fields for ordinary YaRN.
    reference_parameters.pop("mscale")
    reference_parameters.pop("mscale_all_dim")
    SpeculativeConfig.hf_config_override(config)
    corrected_parameters = dict(_rope_parameters(config))
    reference_parameters["rope_theta"] = corrected_parameters["rope_theta"] = 50000.0

    # Isolate get_rope's global object cache and keep this test CPU-only.
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    with set_current_vllm_config(VllmConfig()), torch.device("cpu"):
        reference = rotary_embedding.get_rope(
            head_size=8,
            max_position=262144,
            rope_parameters=reference_parameters,
            dtype=torch.float32,
        )
        corrected = rotary_embedding.get_rope(
            head_size=8,
            max_position=262144,
            rope_parameters=corrected_parameters,
            dtype=torch.float32,
        )

    assert corrected.mscale == pytest.approx(1.4158883083359672)
    torch.testing.assert_close(corrected.cos_sin_cache, reference.cos_sin_cache, rtol=0, atol=0)


def test_kimi_dflash_preserves_composed_user_override():
    config = Qwen3Config(**_kimi_dflash_config_dict())

    def user_override(draft_config):
        rope = _rope_parameters(draft_config)
        rope["attention_factor"] = 1.0
        return draft_config

    override = SpeculativeConfig.compose_draft_hf_overrides(user_override)
    normalized = override(config)

    assert normalized is config
    assert _rope_parameters(normalized)["attention_factor"] == 1.0
