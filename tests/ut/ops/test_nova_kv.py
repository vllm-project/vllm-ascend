# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import nova_kv as attention
from vllm_ascend.ops import nova_kv as ops
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec


def test_cache_interface_preserves_codec_packets_and_padding_slots(monkeypatch):
    identity = torch.eye(8)[None]
    codec = NovaCodec(torch.zeros(1, 8), identity, identity, identity, torch.randn(1, 2, 256, 4))
    key, value = torch.randn(3, 1, 8), torch.randn(3, 1, 8)
    expected = codec.encode(key, value, chunk_size=2)
    slots = torch.tensor([0, 129, -1], dtype=torch.int32)
    calls = []
    monkeypatch.setattr(ops, "write_planar_packets", lambda *args: calls.append(args))
    ops.nova_quant_cache(key, value, codec, (), slots, chunk_size=2)
    assert len(calls) == 1
    assert calls[0][1].dtype == torch.int64
    assert calls[0][1].tolist() == [0, 129, -1]
    for actual, reference in zip(calls[0][2:], expected):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("old_length", [0, 15])
def test_prefill_dispatch_does_not_write_window(monkeypatch, old_length):
    query, key, value = torch.randn(3, 1, 2, 256), torch.randn(3, 1, 256), torch.randn(3, 1, 256)
    window = SimpleNamespace(key="key-window", value="value-window", sink=2, ring=3)
    expected = torch.randn_like(query)
    calls = []
    if old_length == 0:

        def fia(q, k, v, **kwargs):
            calls.append(kwargs)
            assert q.shape == (1, 2, 3, 256)
            assert k.shape == v.shape == (1, 1, 3, 256)
            return expected.permute(1, 2, 0, 3).reshape(1, 2, 3, 256), None

        monkeypatch.setattr(ops.torch_npu, "npu_fused_infer_attention_score", fia, raising=False)
    else:
        monkeypatch.setattr(ops, "planar_cold_reader", lambda *args: "cold")
        monkeypatch.setattr(ops, "exact_window_reader", lambda *args: "exact")

        def bounded(*args, **kwargs):
            calls.append((args, kwargs))
            assert args[3:] == (15, "cold", "exact", 2, 2)
            return expected, None

        monkeypatch.setattr(ops, "bounded_fia_prefill", bounded)
    output, _ = ops.nova_prefill_attention(
        query,
        None,
        (None, None, None),
        None,
        window,
        1 / 16,
        key=key,
        value=value,
        old_length=old_length,
        request_slot=0,
        causal_mask=torch.ones(3, 3, dtype=torch.bool).triu(1),
        query_tile=2,
        kv_tile=128,
    )
    torch.testing.assert_close(output, expected)
    assert len(calls) == 1
    if old_length == 0:
        assert calls[0]["sparse_mode"] == 3


def test_window_update_rejects_mixed_metadata():
    calls = []
    window = SimpleNamespace(write_rows=lambda *args: calls.append(args), seed_chunk=lambda *args: calls.append(args))
    key, value = torch.ones(1, 1, 256), torch.zeros(1, 1, 256)
    slots, positions = torch.tensor([0]), torch.tensor([7])
    ops.nova_exact_window_update(window, key, value, request_slots=slots, positions=positions)
    ops.nova_exact_window_update(window, key, value, request_slot=0, old_length=7)
    assert calls == [(key, value, slots, positions), (key, value, 0, 7)]
    with pytest.raises(ValueError, match="prefill window metadata"):
        ops.nova_exact_window_update(window, key, value, request_slot=0, old_length=7, positions=positions)


def test_mixed_prefill_keeps_request_slots_and_uses_subclass_decode(monkeypatch):
    impl = object.__new__(attention.NovaAttentionImpl)
    impl.num_kv_heads, impl.head_size, impl._limit, impl.scale = 1, 256, 4, 1 / 16
    impl._prefill_query_tile, impl._prefill_kv_tile = 8192, 32768
    impl.nova_codec, impl._window, impl._causal_mask = None, None, None
    query, key = torch.ones(4, 2, 256), torch.ones(4, 1, 256)
    output = torch.empty_like(query)
    calls = []

    def decode(q, k, v, views, metadata, out, codec, request_slots):
        assert metadata.seq_lens_list == [20]
        assert request_slots.tolist() == [2]
        calls.append("decode")
        out.fill_(7)

    def prefill(q, *args, **kwargs):
        assert kwargs["old_length"] == 0 and kwargs["request_slot"] == 1
        calls.append("prefill")
        return torch.full_like(q, 9), None

    impl._decode = decode
    monkeypatch.setattr(attention, "nova_prefill_attention", prefill)
    monkeypatch.setattr(attention, "nova_exact_window_update", lambda *args, **kwargs: calls.append("window"))
    metadata = SimpleNamespace(
        seq_lens_list=[20, 3],
        nova_request_slots_cpu=[2, 1],
        actual_seq_lengths_q=[1, 4],
        seq_lens_gpu=torch.tensor([20, 3]),
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        nova_request_slots=torch.tensor([2, 1]),
    )
    impl._prefill(query, key, key, (), metadata, output)
    assert calls == ["decode", "prefill", "window"]
    torch.testing.assert_close(output[0], torch.full_like(output[0], 7))
    torch.testing.assert_close(output[1:], torch.full_like(output[1:], 9))
