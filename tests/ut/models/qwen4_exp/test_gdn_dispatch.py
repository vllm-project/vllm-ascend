# SPDX-License-Identifier: Apache-2.0
"""Qwen decoder construction must reach Ascend GDN without changing weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import model
from vllm_ascend.ops import gdn


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_decoder_uses_ascend_gdn_with_qwen_projection_initialization(monkeypatch, sequence_parallel):
    config = SimpleNamespace(ple_layer_ids=[], hc_count=4, hidden_size=2, hc_lowrank=2, rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sequence_parallel),
    )
    calls = []

    def qwen_init(self, config, vllm_config, prefix, gqa_interleaved_layout, reduce_results):
        torch.nn.Module.__init__(self)
        self.in_proj_qkvz = torch.nn.Linear(2, 2, bias=False)
        calls.append((config, vllm_config, prefix, gqa_interleaved_layout, reduce_results))

    def ascend_forward(self, hidden_states):
        return hidden_states + 1

    def unexpected_upstream_forward(*args, **kwargs):
        raise AssertionError("Qwen4Exp must execute Ascend GDN, not upstream CUDA GDN")

    monkeypatch.setattr(model.upstream_model.QwenGatedDeltaNetAttention, "__init__", qwen_init)
    monkeypatch.setattr(model.upstream_model.QwenGatedDeltaNetAttention, "forward", unexpected_upstream_forward)
    monkeypatch.setattr(gdn.AscendGatedDeltaNetAttention, "forward", ascend_forward)
    monkeypatch.setattr(model, "AscendQwen4ExpSparseMoeBlock", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model.upstream_model, "HyperConnectionConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(model.upstream_model, "GatedResidual", lambda *args, **kwargs: torch.nn.Identity())

    decoder = model.AscendQwen4ExpDecoderLayer(
        vllm_config=vllm_config, layer_type="linear_attention", prefix="model.layers.0"
    )
    attention = decoder.linear_attn
    hidden_states = torch.zeros(2, 2)
    torch.testing.assert_close(attention(hidden_states), torch.ones_like(hidden_states))
    assert calls == [(config, vllm_config, "model.layers.0.linear_attn", False, not sequence_parallel)]
    assert set(decoder.state_dict()) == {"linear_attn.in_proj_qkvz.weight"}
    assert type(attention)._forward_core is gdn.AscendGatedDeltaNetAttention._forward_core
    assert attention.get_attn_backend() is gdn.AscendGDNFusedAttentionBackend
