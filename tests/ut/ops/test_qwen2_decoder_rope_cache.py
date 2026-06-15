import torch

from vllm_ascend.ops import qwen2_decoder


def test_optimized_apply_rotary_pos_emb_by_cache_flattens_batched_positions(monkeypatch):
    calls = []

    def fake_rotary_mul_by_cache(x, positions, rotary_emb, **kwargs):
        calls.append((x.clone(), positions.clone(), rotary_emb, kwargs))
        return x + len(calls)

    monkeypatch.setattr(qwen2_decoder, "rotary_mul_by_cache", fake_rotary_mul_by_cache)

    q = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).view(2, 3, 4, 6)
    k = torch.arange(2 * 2 * 4 * 6, dtype=torch.float32).view(2, 2, 4, 6)
    position_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    rotary_emb = object()

    q_out, k_out = qwen2_decoder.optimized_apply_rotary_pos_emb_by_cache(q, k, position_ids, rotary_emb)

    q_tokens = q.transpose(1, 2).contiguous().view(8, 3, 1, 6)
    k_tokens = k.transpose(1, 2).contiguous().view(8, 2, 1, 6)
    expected_q = (q_tokens + 1).view(2, 4, 3, 6).transpose(1, 2).contiguous()
    expected_k = (k_tokens + 2).view(2, 4, 2, 6).transpose(1, 2).contiguous()

    assert torch.equal(q_out, expected_q)
    assert torch.equal(k_out, expected_k)
    assert torch.equal(calls[0][1], position_ids.reshape(-1))
    assert torch.equal(calls[1][1], position_ids.reshape(-1))
    assert calls[0][2] is rotary_emb
    assert calls[1][2] is rotary_emb
    assert calls[0][3]["layout"] == "T11D"
    assert calls[1][3]["layout"] == "T11D"
