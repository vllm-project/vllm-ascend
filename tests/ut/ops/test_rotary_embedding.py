import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AttentionV1  # noqa: F401


class DummyAscendConfig:

    class torchair_graph_config:
        enabled = True


@pytest.fixture(autouse=True)
def patch_ascend_config(monkeypatch):
    monkeypatch.setattr(
        "my_module.attention.attention_v1.get_ascend_config",
        lambda: DummyAscendConfig,
    )
    yield


@pytest.fixture(autouse=True)
def patch_npu_apply(monkeypatch):
    # stub npu_apply_rotary_pos_emb: q->q+1, k->k+2
    def fake_apply(q, k, cos, sin):
        return q + 1.0, k + 2.0

    monkeypatch.setattr(torch_npu, "npu_apply_rotary_pos_emb", fake_apply)
    yield


@pytest.fixture(autouse=True)
def patch_set_cos_sin_cache(monkeypatch):
    monkeypatch.setattr("my_module.attention.attention_v1.__set_cos_sin_cache",
                        lambda *args, **kwargs: None)
    yield


def test_rope_forward_basic():
    attn = AttentionV1.__new__(AttentionV1)
    attn.max_position_embeddings = 1024
    attn.head_size = 2
    # pre-allocate cos/sin tensors
    attn.cos = torch.randn(1, 3)
    attn.sin = torch.randn(1, 3)
    attn.cos_embed = None
    attn.sin_embed = None

    # input tensor: batch=1, seq_len=3, embed_dim=num_heads*head_size=2
    batch, seq_len, embed_dim = 1, 3, 2
    positions_ids = torch.randint(0, seq_len, (batch, seq_len))
    query = torch.arange(batch * seq_len * embed_dim,
                         dtype=torch.float32).view(batch, seq_len, embed_dim)
    key = torch.arange(batch * seq_len * embed_dim, dtype=torch.float32).view(
        batch, seq_len, embed_dim) + 100.0

    q_out, k_out = attn.rope_forward(
        positions_ids=positions_ids,
        query=query,
        key=key,
        offsets=None,
        max_seq_len=None,
    )

    # raw query/key reshape to (..., num_heads, head_size)
    # unsqueeze(1)，fake_apply+1/+2, flatten(-2), combine last 2 dims
    q_view = query.view(batch, seq_len // 1, 1, attn.head_size)  # num_heads=1
    k_view = key.view(batch, seq_len // 1, 1, attn.head_size)
    q_unsq = q_view.unsqueeze(1)
    k_unsq = k_view.unsqueeze(1)

    expected_q = (q_unsq + 1.0).flatten(-2)
    expected_k = (k_unsq + 2.0).flatten(-2)

    assert q_out.shape == expected_q.shape
    assert k_out.shape == expected_k.shape

    assert torch.allclose(q_out, expected_q)
    assert torch.allclose(k_out, expected_k)
