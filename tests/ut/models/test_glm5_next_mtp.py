# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
import torch
import vllm.config.speculative as speculative_config
from torch import nn

from vllm_ascend.models.glm5next.config import Glm5NextTextConfig
from vllm_ascend.models.glm5next.model import get_spec_layer_idx_from_weight_name
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
    assert mtp._rewrite_spec_layer_name(45, "layers.45.embed_tokens.weight") == "model.embed_tokens.weight"


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


def _make_mtp_loader(params):
    mtp = object.__new__(Glm5NextMTP)
    nn.Module.__init__(mtp)
    mtp.config = SimpleNamespace(
        n_routed_experts=None,
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        mla_nope=False,
        qk_rope_head_dim=0,
    )
    mtp.model = SimpleNamespace(
        mtp_start_layer_idx=45,
        num_mtp_layers=1,
    )
    mtp.named_parameters = lambda: params.items()
    return mtp


def test_mtp_loader_accepts_loader_kwargs_and_skips_rotary_caches():
    weight_loader = MagicMock()
    param_name = "model.layers.45.enorm.weight"
    param = SimpleNamespace(weight_loader=weight_loader)
    mtp = _make_mtp_loader({param_name: param})
    loaded_weight = torch.ones(4)
    weights = [
        ("model.layers.45.rotary_emb.cos_cached", torch.ones(1)),
        ("model.layers.45.rotary_emb.sin_cached", torch.ones(1)),
        (param_name, loaded_weight, {"load_metadata": "mtp"}),
    ]

    loaded = mtp.load_weights(weights)

    assert loaded == {param_name}
    weight_loader.assert_called_once_with(
        param,
        loaded_weight,
        load_metadata="mtp",
    )
    assert mtp.has_own_lm_head is False


def test_mtp_loader_rejects_missing_fused_projection_target():
    mtp = _make_mtp_loader({})
    expected_name = (
        "model.layers.45.mtp_block.self_attn."
        "fused_qkv_a_proj.weight"
    )

    with pytest.raises(KeyError, match=expected_name):
        mtp.load_weights(
            [
                (
                    "model.layers.45.self_attn.q_a_proj.weight",
                    torch.randn(4, 4),
                )
            ]
        )
