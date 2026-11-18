# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next import model as model_module
from vllm_ascend.models.glm5next import mtp as mtp_module

TP_SIZE = 2
HIDDEN_SIZE = 4


class _Norm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return normalized if residual is None else (normalized, x)


class _Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)

    def forward(self, hidden_states, positions):
        assert hidden_states.shape[0] == positions.shape[0]
        self.hidden_states = hidden_states
        # Each TP rank contributes the same partial projection in this mock.
        return hidden_states * (TP_SIZE if self.o_proj.reduce_results else 1)


class _MoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.already_sequence_parallel = None

    def forward(self, x, already_sequence_parallel=False):
        self.already_sequence_parallel = already_sequence_parallel
        return x * 0.5


@pytest.mark.parametrize("use_sp", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 3, 4])
@pytest.mark.parametrize("rank", range(TP_SIZE))
def test_mtp_sp_preserves_full_recycled_states(monkeypatch, use_sp, num_tokens, rank):
    monkeypatch.setattr(model_module, "Glm5NextMLAAttention", _Attention)
    monkeypatch.setattr(model_module, "Glm5NextMoE", _MoE)
    monkeypatch.setattr(model_module, "RMSNorm", _Norm)
    config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=HIDDEN_SIZE,
        mhc=True,
        n_routed_experts=8,
        rms_norm_eps=1e-6,
        v_head_dim=HIDDEN_SIZE,
        kv_lora_rank=HIDDEN_SIZE,
        num_attention_heads=TP_SIZE,
        qk_nope_head_dim=HIDDEN_SIZE,
        qk_rope_head_dim=HIDDEN_SIZE,
        q_lora_rank=HIDDEN_SIZE,
        max_position_embeddings=128,
        mla_use_nope=False,
        mlp_layer_types=["sparse"],
    )
    runtime = SimpleNamespace(
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=use_sp),
    )
    decoder = model_module.Glm5NextDecoderLayer(runtime, config, 1, is_mtp_layer=True)
    assert decoder.self_attn.o_proj.reduce_results is (not use_sp)

    predictor = mtp_module.Glm5NextMultiTokenPredictorLayer.__new__(mtp_module.Glm5NextMultiTokenPredictorLayer)
    nn.Module.__init__(predictor)
    predictor.enorm = _Norm()
    predictor.hnorm = _Norm()
    predictor.eh_proj = nn.Identity()
    predictor.mtp_block = decoder
    predictor.shared_head = nn.Module()
    predictor.shared_head.norm = _Norm()
    # Isolate the unrelated fused input normalization kernel.
    monkeypatch.setattr(torch.ops.vllm, "glm5next_mtp_eh_norm", lambda p, e, h, *args: e + h)

    collectives: list[str] = []

    def pad(x):
        return torch.nn.functional.pad(x, (0, 0, 0, (-x.shape[0]) % TP_SIZE))

    def shard(x):
        return x.chunk(TP_SIZE, dim=0)[rank]

    def reduce_scatter(x):
        assert collectives[-1] == "ag_attention"
        collectives.append("rs")
        return shard(x * TP_SIZE)

    def all_gather(x):
        if x.shape[-1] == HIDDEN_SIZE:
            collectives.append("ag_attention")
            full_states = norm(combined)
        else:
            assert collectives[-1] == "rs"
            collectives.append("ag_output")
            full_states = torch.cat([normalized * 0.5, residual], dim=-1)
        torch.testing.assert_close(x, shard(pad(full_states)))
        return pad(full_states)

    monkeypatch.setattr(model_module, "pad_sequence_parallel_tokens", pad)
    monkeypatch.setattr(model_module, "sp_shard", shard)
    monkeypatch.setattr(model_module, "sp_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(model_module, "sp_all_gather", all_gather)
    # A second chunk/gather inside MoE would corrupt the rank-local token layout.
    monkeypatch.setattr(model_module, "sequence_parallel_chunk", lambda x: pytest.fail("double sharding"))
    monkeypatch.setattr(model_module, "tensor_model_parallel_all_gather", lambda *args: pytest.fail("early gather"))

    embeds = torch.arange(num_tokens * HIDDEN_SIZE, dtype=torch.float32).view(num_tokens, HIDDEN_SIZE) / 10
    hidden = torch.flip(embeds, dims=[-1])
    original_embeds = embeds.clone()
    norm = _Norm()
    for step in range(3):
        combined = embeds + hidden
        normalized, residual = norm(norm(combined) * TP_SIZE, combined)
        expected, _ = norm(normalized * 0.5, residual)
        logits_hidden, recycled_hidden = predictor(
            torch.arange(num_tokens), torch.arange(num_tokens) + step, hidden, embeds, step
        )
        for output in (logits_hidden, recycled_hidden):
            assert output.shape == (num_tokens, HIDDEN_SIZE)
            torch.testing.assert_close(output, expected)
        assert decoder.mlp.already_sequence_parallel is use_sp
        torch.testing.assert_close(decoder.self_attn.hidden_states, norm(combined))
        hidden = recycled_hidden
    expected_collectives = ["ag_attention", "rs", "ag_output"] * 3 if use_sp else []
    assert collectives == expected_collectives
    torch.testing.assert_close(embeds, original_embeds)
