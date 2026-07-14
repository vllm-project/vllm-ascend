# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV4Model,
    _is_dspark_target_layer,
)
from vllm_ascend.models.deepseek_v4_draft import (
    is_deepseek_v4_dspark_config,
    remap_dspark_mtp_weight_name,
)
from vllm_ascend.patch.platform.patch_speculative_config import hf_config_override


def _make_dspark_config() -> PretrainedConfig:
    return PretrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=43,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
    )


def test_dspark_config_detection():
    config = _make_dspark_config()

    assert is_deepseek_v4_dspark_config(config)
    assert not is_deepseek_v4_dspark_config(SimpleNamespace(model_type="deepseek_v4", dspark_block_size=0))


def test_dspark_speculative_config_override_exposes_draft_contract():
    config = hf_config_override(_make_dspark_config())

    assert config.model_type == "deepseek_mtp"
    assert config.architectures == ["DeepSeekV4DSparkMTPModel"]
    assert not hasattr(config, "n_predict")
    assert config.ptd_token_id == 128799
    assert config.dspark_num_mtp_layers == 3
    assert not hasattr(config, "eagle_aux_hidden_state_layer_ids")


def test_non_dspark_deepseek_v4_keeps_existing_mtp_override():
    config = PretrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_nextn_predict_layers=1,
    )

    overridden = hf_config_override(config)

    assert overridden.architectures == ["DeepSeekV4MTPModel"]
    assert not hasattr(overridden, "eagle_aux_hidden_state_layer_ids")


@pytest.mark.parametrize(
    ("source_name", "expected_name"),
    [
        ("mtp.0.main_proj.weight", "model.layers.43.main_proj.weight"),
        ("mtp.0.main_norm.weight", "model.layers.43.main_norm.weight"),
        ("mtp.0.attn.attn_sink", "model.layers.43.self_attn.attn_sink"),
        ("mtp.0.attn.wq_a.weight", "model.layers.43.self_attn.wq_a.weight"),
        (
            "mtp.1.ffn.shared_experts.w1.weight",
            "model.layers.44.mlp.shared_experts.gate_proj.weight",
        ),
        (
            "mtp.1.ffn.shared_experts.w2.weight",
            "model.layers.44.mlp.shared_experts.down_proj.weight",
        ),
        (
            "mtp.1.ffn.shared_experts.w3.weight",
            "model.layers.44.mlp.shared_experts.up_proj.weight",
        ),
        (
            "mtp.1.ffn.gate.bias",
            "model.layers.44.mlp.gate.e_score_correction_bias",
        ),
        ("mtp.2.hc_head_fn", "model.layers.45.hc_head_fn"),
        (
            "mtp.2.markov_head.markov_w2.weight",
            "model.layers.45.markov_head.markov_w2.weight",
        ),
    ],
)
def test_remap_dspark_mtp_weight_name(source_name: str, expected_name: str):
    assert remap_dspark_mtp_weight_name(source_name, num_hidden_layers=43) == expected_name


def test_remap_dspark_mtp_weight_name_ignores_non_draft_weights():
    assert remap_dspark_mtp_weight_name("model.layers.0.attn.wq_a.weight", 43) is None
    assert remap_dspark_mtp_weight_name("mtp.2.confidence_head.proj.weight", 43) is None


def test_dspark_target_hidden_states_use_dedicated_buffer():
    wrapper = AscendDeepseekV4ForCausalLM.__new__(AscendDeepseekV4ForCausalLM)
    nn.Module.__init__(wrapper)
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model._mtp_hidden_buffer = torch.zeros((2, 16))
    model._dspark_target_layer_ids = (40, 41, 42)
    model._dspark_hidden_buffer = torch.ones((2, 12))
    wrapper.model = model

    assert wrapper.get_mtp_target_hidden_states() is model._dspark_hidden_buffer
    model._dspark_target_layer_ids = ()
    assert wrapper.get_mtp_target_hidden_states() is model._mtp_hidden_buffer


def test_dspark_target_layer_ids_use_checkpoint_semantics():
    target_layer_ids = {40, 41, 42}

    assert not _is_dspark_target_layer(39, target_layer_ids)
    assert _is_dspark_target_layer(40, target_layer_ids)
    assert _is_dspark_target_layer(41, target_layer_ids)
    assert _is_dspark_target_layer(42, target_layer_ids)
