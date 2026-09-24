# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram


@pytest.mark.parametrize("tokens", [1, 7])
@pytest.mark.parametrize("rotated", [False, True])
def test_gate_matches_original_basis(tokens, rotated):
    torch.manual_seed(41)
    hidden = torch.randn(tokens, 4, 64)
    key = torch.randn_like(hidden)
    value = torch.randn(tokens, 64)
    q, k = torch.randn(2, 4, 64)
    mask = torch.arange(tokens) % 3 != 1
    eps = 1e-20
    query = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps) * q
    normalized_key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + eps) * k
    dot = (query * normalized_key).sum(-1) / 8
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    expected = hidden + gate.masked_fill(~mask[:, None], 0)[..., None] * value[:, None]
    rotation = None
    if rotated:
        rotation = torch.linalg.qr(torch.randn(32, 32)).Q
        hidden = (hidden.unflatten(-1, (2, 32)) @ rotation).flatten(-2)
        value = (value.unflatten(-1, (2, 32)) @ rotation).flatten(-2)
        expected = (expected.unflatten(-1, (2, 32)) @ rotation).flatten(-2)
    actual = engram_gate(hidden, key, value, q * k, rotation, mask, eps)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(actual[~mask], hidden[~mask])


def test_bf16_wkv_gate_without_rotation():
    torch.manual_seed(42)
    config = SimpleNamespace(
        hidden_size=64, hc_mult=4, rms_norm_eps=1e-20, engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32
    )
    layer = AscendEngram(config)
    with torch.no_grad():
        layer.q_weight.normal_()
        layer.k_weight.normal_()
    hidden = torch.randn(3, 4, 64).bfloat16()
    rows = torch.randn(3, 128).bfloat16()
    mask = torch.tensor([True, False, True])
    kv = torch.nn.functional.linear(rows, layer.wkv.weight).float()
    key, value = kv.split([256, 64], -1)
    key = key.view_as(hidden)
    query = torch.nn.functional.rms_norm(hidden.float(), (64,), eps=config.rms_norm_eps) * layer.q_weight.float()
    normalized_key = torch.nn.functional.rms_norm(key, (64,), eps=config.rms_norm_eps) * layer.k_weight.float()
    dot = (query * normalized_key).sum(-1) / 8
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask[:, None], 0)
    expected = (hidden.float() + gate[..., None] * value[:, None]).bfloat16()
    torch.testing.assert_close(layer(hidden, rows, mask, None), expected, atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_bf16_checkpoint_preserves_rows(tmp_path, monkeypatch, rank):
    key = "layers.1.engram.embed.weight"
    source = torch.linspace(-1.37, 2.41, 19 * 64).reshape(19, 64).bfloat16()
    save_file({key: source}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors"}}))
    monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(embedding_mod, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: None)
    monkeypatch.setattr(
        embedding_mod.AscendParallelEngramEmbedding,
        "_allocate_weights",
        lambda self: (torch.zeros(self.part_num_embeddings, 64, dtype=self.storage_dtype), None),
    )
    table = embedding_mod.AscendParallelEngramEmbedding(19, 64, (9, 10), 0, storage_dtype=torch.bfloat16)
    table.bind_checkpoint(tmp_path, key)
    table.load_checkpoint(tmp_path, key, chunk_rows=7)
    assert table.weight_scale_inv is None
    start, end = table.vocab_start_idx, table.vocab_end_idx
    assert torch.equal(table.weight, source[start:end])
    ids = torch.tensor([[start], [end - 1], [-1], [end], [start + 7]]).repeat(1, 2)
    out = torch.empty(5, 1, 64, dtype=torch.bfloat16)
    table.lookup(ids, out)
    expected = source[torch.tensor([start, end - 1, 0, 0, start + 7])].clone()
    expected[2:4] = 0
    assert torch.equal(out[:, 0], expected)
