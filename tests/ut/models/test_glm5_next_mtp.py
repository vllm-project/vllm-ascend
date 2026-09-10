# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import get_args

import torch
import vllm.config.speculative as speculative_config
from torch import nn

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next.config import Glm5NextTextConfig
from vllm_ascend.models.glm5next.model import (
    Glm5NextForConditionalGeneration,
    _mark_zero_initialized_rms_norm_biases,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP
from vllm_ascend.patch.platform.patch_speculative_config import (
    _normalize_legacy_qwen3_dspark_config,
)


def test_get_spec_layer_idx_accepts_checkpoint_prefixes():
    config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, "model.layers.45.enorm.weight") == 45
    assert get_spec_layer_idx_from_weight_name(config, "layers.46.self_attn.q_a_proj.weight") == 46
    assert get_spec_layer_idx_from_weight_name(config, "model.layers.44.mlp.weight") is None
    assert get_spec_layer_idx_from_weight_name(config, "rot.weight") is None


def test_mtp_rewrites_layer_and_shared_weight_names():
    mtp = object.__new__(Glm5NextMTP)

    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.self_attn.q_a_proj.weight",
        )
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.shared_head.norm.weight",
        )
        == "model.layers.45.shared_head.norm.weight"
    )


def test_multimodal_mapper_flattens_modelslim_forget_gate_prefix():
    weight_name = "model.language_model.layers.0.self_attn.forget_gate.f_b_proj.weight"

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name(weight_name)
        == "language_model.model.layers.0.self_attn.f_b_proj.weight"
    )

    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.attn_hc.fn")
        == "language_model.model.layers.1.hc_attn_fn"
    )
    assert (
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper._map_name("model.language_model.layers.1.ffn_hc.scale")
        == "language_model.model.layers.1.hc_ffn_scale"
    )


def test_synthetic_rms_norm_biases_are_zeroed_and_marked(monkeypatch):
    class FakeRMSNorm(nn.Module):
        def __init__(self, *, bias_loaded: bool):
            super().__init__()
            self.bias = nn.Parameter(torch.ones(2), requires_grad=False)
            self.bias_loaded = bias_loaded

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.synthetic = FakeRMSNorm(bias_loaded=False)
            self.checkpoint = FakeRMSNorm(bias_loaded=True)

    monkeypatch.setattr(glm5next_model, "RMSNorm", FakeRMSNorm)
    model = TestModel()
    loaded_params: set[str] = set()

    _mark_zero_initialized_rms_norm_biases(model, loaded_params)

    assert loaded_params == {"synthetic.bias"}
    assert torch.equal(model.synthetic.bias, torch.zeros(2))
    assert torch.equal(model.checkpoint.bias, torch.ones(2))


def test_glm5_speculative_config_selects_mtp_architecture():
    config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        num_nextn_predict_layers=2,
    )

    result = _normalize_legacy_qwen3_dspark_config(config)

    assert result.model_type == "glm5_next_mtp"
    assert result.n_predict == 2
    assert result.architectures == ["Glm5NextMTPModel"]
    assert "glm5_next_mtp" in get_args(speculative_config.MTPModelTypes)
