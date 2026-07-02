from types import SimpleNamespace

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


def test_flatten_qwen2_positions_accepts_already_flattened_1d():
    position_ids = torch.arange(8)

    positions = qwen2_decoder._flatten_qwen2_positions(position_ids, batch_size=2, seq_len=4)

    assert torch.equal(positions, position_ids)


def test_qwen2_rotary_cache_rebuilds_to_ref_dtype():
    rotary = qwen2_decoder.AscendQwen2RotaryEmbedding.__new__(qwen2_decoder.AscendQwen2RotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.inv_freq = torch.tensor([1.0, 0.5])
    rotary.attention_scaling = 1.0
    rotary.max_seq_len_cached = 2
    rotary._dynamic_frequency_update = None

    rotary._set_cos_sin_cache(2, device=torch.device("cpu"), dtype=torch.float32)
    assert rotary.cos_sin_cache.dtype == torch.float32

    ref_tensor = torch.empty(1, dtype=torch.float16)
    rotary.ensure_cos_sin_cache(torch.tensor([[0, 1]], dtype=torch.long), ref_tensor)

    assert rotary.cos_sin_cache.dtype == torch.float16
    assert rotary.cos_sin_cache.device == ref_tensor.device


def test_cache_update_uses_rope_kwargs_detection():
    class DynamicLike:
        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            return key_states, value_states

    class SinkLike:
        def update(self, key_states, value_states, layer_idx, cache_kwargs):
            return cache_kwargs.get("sin"), cache_kwargs.get("cos")

    assert not qwen2_decoder._cache_update_uses_rope_kwargs(DynamicLike())
    assert qwen2_decoder._cache_update_uses_rope_kwargs(SinkLike())


def test_materialize_qwen2_cache_update_rope_matches_ref_dtype():
    rotary = SimpleNamespace(
        cos_sin_cache=torch.tensor([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]),
        is_neox_style=True,
    )
    rotary._match_cos_sin_cache_dtype = lambda ref_tensor: rotary.cos_sin_cache.to(dtype=ref_tensor.dtype)

    ref_tensor = torch.empty(1, dtype=torch.float16)
    cos, sin = qwen2_decoder._materialize_qwen2_cache_update_rope(
        torch.tensor([[0, 1]], dtype=torch.long),
        rotary,
        ref_tensor,
    )

    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16
    expected_cos = torch.tensor(
        [[[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]],
        dtype=torch.float16,
    )
    expected_sin = torch.tensor(
        [[[10.0, 20.0, 10.0, 20.0], [30.0, 40.0, 30.0, 40.0]]],
        dtype=torch.float16,
    )
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)
