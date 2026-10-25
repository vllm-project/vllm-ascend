# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise decoder construction with the real Transformers config schema."""

from types import SimpleNamespace

import pytest
from torch import nn
from transformers import Glm5NextTextConfig
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.glm5next import model as glm_model


class _Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.o_proj = SimpleNamespace(reduce_results=True)


class _KDA(_Attention):
    pass


class _MLA(_Attention):
    pass


class _Dense(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class _MoE(_Dense):
    pass


@pytest.mark.parametrize(
    "layer_idx,is_mtp,attention,mlp",
    [(0, False, _KDA, _Dense), (3, False, _MLA, _MoE), (44, False, _KDA, _MoE), (45, True, _MLA, _MoE)],
)
def test_decoder_uses_transformers_fields(monkeypatch, layer_idx, is_mtp, attention, mlp):
    config = Glm5NextTextConfig(hidden_size=8, mhc=True, mla_use_nope=True, hc_mult=2, hc_sinkhorn_iters=7)
    # The draft path must remain MLA even if a config lists KDA at its index.
    config.layer_types.append("linear_attention")
    assert not hasattr(config, "is_moe")
    assert not hasattr(config, "is_kda_layer")
    monkeypatch.setattr(glm_model, "Glm5NextLinearAttention", _KDA)
    monkeypatch.setattr(glm_model, "Glm5NextMLAAttention", _MLA)
    monkeypatch.setattr(glm_model, "Glm5NextMLP", _Dense)
    monkeypatch.setattr(glm_model, "Glm5NextMoE", _MoE)
    vllm_config = SimpleNamespace(
        cache_config=None, quant_config=None, parallel_config=SimpleNamespace(use_sequence_parallel_moe=False)
    )

    with set_current_vllm_config(VllmConfig()):
        layer = glm_model.Glm5NextDecoderLayer(vllm_config, config, layer_idx, is_mtp_layer=is_mtp)

    assert layer.is_moe
    assert type(layer.self_attn) is attention
    assert type(layer.mlp) is mlp
    if attention is _MLA:
        assert layer.self_attn.kwargs["skip_rope"] is True
    if not is_mtp:
        assert layer.n == 2
        assert layer.mhc_sinkhorn_iterations == 7
        assert layer.hc_attn_fn.shape == (8, 16)
        assert layer.mhc_tau == 0.05
        assert layer.mhc_post_mult_value == 2.0


def test_unsupported_kpool_tail_is_rejected():
    config = Glm5NextTextConfig(index_kpool_always_select_tail=False)
    with pytest.raises(NotImplementedError, match="index_kpool_always_select_tail"):
        glm_model._validate_supported_config(config)
