# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from torch import nn

import vllm_ascend.models.glm5next.attention as attention


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_sparse_indexer_only_constructs_nonempty_rope(monkeypatch, rope_dim):
    calls = []

    def fake_rope(dim, **kwargs):
        assert dim > 0, "Zero-width rotary cache must never be constructed"
        calls.append(dim)
        return nn.Identity()

    def fake_wrapper(*args, **kwargs):
        module = nn.Identity()
        module.mla_attn = SimpleNamespace()
        return module

    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(attention, "get_rope", fake_rope)
    for name in ("DeepSeekV2FusedQkvAProjLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm", "Indexer"):
        monkeypatch.setattr(attention, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(attention, "MultiHeadLatentAttentionWrapper", fake_wrapper)
    model = attention.Glm5NextMLAAttention(
        vllm_config=SimpleNamespace(),
        config=SimpleNamespace(index_topk=2048, rope_parameters=None, indexer_rope_interleave=True, rms_norm_eps=1e-6),
        hidden_size=32,
        num_heads=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        q_lora_rank=16,
        kv_lora_rank=8,
        skip_rope=True,
    )
    assert calls == ([64] if rope_dim else [])
    assert (model.indexer_rope_emb is None) == (rope_dim == 0)
