# SPDX-License-Identifier: Apache-2.0
"""NOVA-KV lifecycle, cache writes and graph replay against reconstructed KV."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tilelang")
pytest.importorskip("torch_npu")

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.nova_kv import NovaAttentionImpl
from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.nova_kv.cache import ExactWindowPool, planar_cache_views, write_planar_packets
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec

pytestmark = pytest.mark.skipif(not torch.npu.is_available() or not is_950(), reason="requires Ascend 950")


@pytest.mark.parametrize("batch", [1, 4])
def test_full_attention_replay_keeps_exact_windows_and_padded_rows(batch):
    torch.manual_seed(319)
    device, heads, groups, dim, capacity = "npu:0", 1, 6, 256, 768
    identity = torch.eye(dim, device=device)[None]
    codec = NovaCodec(
        torch.randn(heads, dim, device=device) * 0.1,
        identity,
        identity,
        identity,
        torch.randn(heads, 64, 256, 4, device=device).half(),
    )
    history_key = torch.randn(batch, capacity, heads, dim, device=device, dtype=torch.bfloat16)
    history_value = torch.randn_like(history_key)
    impl = object.__new__(NovaAttentionImpl)
    impl.head_size, impl.num_kv_heads, impl.num_heads = dim, heads, heads * groups
    impl.scale, impl._limit, impl._sink, impl._recent = 1 / 16, batch, 64, 256
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=capacity))
    impl._options = SimpleNamespace(codec_batch_size=64)
    impl.nova_codec = codec
    impl._window = ExactWindowPool(batch, heads, dim, 64, 256, device, torch.bfloat16)
    impl._causal_mask = None
    query = torch.randn(batch, heads * groups, dim, device=device, dtype=torch.bfloat16)
    key, value, output = history_key[:, 0].contiguous(), history_value[:, 0].contiguous(), torch.empty_like(query)
    cache = torch.zeros(batch * 6, 128, heads, 134, dtype=torch.uint8, device=device)
    views = planar_cache_views(cache)
    encoded = codec.encode(history_key.flatten(0, 1), history_value.flatten(0, 1), chunk_size=64)
    write_planar_packets(views, torch.arange(batch * capacity, device=device), *encoded)
    reconstructed = codec.decode(*encoded)
    metadata = SimpleNamespace(
        seq_lens_list=[1] * batch,
        seq_lens_gpu=torch.ones(batch, dtype=torch.int32, device=device),
        block_tables=torch.arange(batch * 6, dtype=torch.int32, device=device).reshape(batch, 6),
        nova_request_slots=torch.full((batch,), -1, dtype=torch.int64, device=device),
        slot_mapping=torch.zeros(batch, dtype=torch.int64, device=device),
        attn_state=AscendAttentionState.DecodeOnly,
    )

    def run():
        return impl.forward(SimpleNamespace(), query, key, value, cache, metadata, output=output)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for step, length in enumerate((333, 513, 700)):
        query.normal_(0, 0.2)
        expected = torch.zeros_like(query)
        for row in range(batch):
            request = (row + step) % batch
            metadata.nova_request_slots[row] = request
            metadata.seq_lens_gpu[row] = length
            metadata.slot_mapping[row] = request * capacity + length - 1
            metadata.block_tables[row].copy_(torch.arange(request * 6, request * 6 + 6, device=device))
            key[row].copy_(history_key[request, length - 1])
            value[row].copy_(history_value[request, length - 1])
            impl._window.seed_chunk(
                history_key[request, : length - 1], history_value[request, : length - 1], request, 0
            )
            dense_key, dense_value = [
                x[request * capacity : request * capacity + length].clone() for x in reconstructed
            ]
            for dense, exact in ((dense_key, history_key), (dense_value, history_value)):
                dense[:64] = exact[request, :64].float()
                dense[-257:] = exact[request, length - 257 : length].float()
            scores = torch.einsum("hgd,thd->hgt", query[row].reshape(heads, groups, dim).float(), dense_key) / 16
            expected[row] = torch.einsum("hgt,thd->hgd", scores.softmax(-1), dense_value).reshape_as(expected[row])
        if batch == 4:
            metadata.nova_request_slots[-1] = -1
            metadata.slot_mapping[-1] = 0
            expected[-1].zero_()
        graph.replay()
        actual = captured.clone()
        eager = run().clone()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0.01, atol=0.003)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
