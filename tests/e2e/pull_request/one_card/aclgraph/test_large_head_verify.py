# SPDX-License-Identifier: Apache-2.0
"""NPU graph precision regression without model downloads.

CPU UTs cover mask construction. This test additionally exercises the real
512-head NPU kernel and graph replay after in-place metadata/input updates.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE


def _reference(query, keys, values, block_tables, lengths, query_len, scale):
    """Use CPU FP32 SDPA on the visible prefix of each individual query."""
    query, keys, values = query.cpu().float(), keys.cpu().float(), values.cpu().float()
    tables = block_tables.cpu().tolist()
    result = torch.empty_like(query)
    block_size = keys.shape[1]
    for request, length in enumerate(lengths):
        for offset in range(query_len):
            row = request * query_len + offset
            visible = length - query_len + offset + 1
            key = torch.stack([keys[tables[request][t // block_size], t % block_size] for t in range(visible)])
            value = torch.stack([values[tables[request][t // block_size], t % block_size] for t in range(visible)])
            result[row] = torch.nn.functional.scaled_dot_product_attention(
                query[row].unsqueeze(1), key.transpose(0, 1), value.transpose(0, 1), enable_gqa=True, scale=scale
            ).squeeze(1)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_large_head_verify_graph_replays_dynamic_metadata(dtype):
    generator = torch.Generator().manual_seed(2026)
    head_size = FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
    query_len = 4
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, head_size
    impl.scale = head_size**-0.5
    impl.key_cache = torch.randn(6, 128, 2, head_size, generator=generator).to(device="npu", dtype=dtype)
    impl.value_cache = torch.randn(6, 128, 2, head_size, generator=generator).to(device="npu", dtype=dtype)
    query = torch.randn(2 * query_len, 4, head_size, generator=generator).to(device="npu", dtype=dtype)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([129, 145], dtype=torch.int32),
        seq_lens_device=torch.tensor([129, 145], dtype=torch.int32, device="npu"),
        block_tables=torch.tensor([[3, 1], [4, 2]], dtype=torch.int32, device="npu"),
    )
    output = torch.empty_like(query)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            impl._forward_large_head_graph_verify_attention(query, metadata, output)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        impl._forward_large_head_graph_verify_attention(query, metadata, output)
    addresses = (query.data_ptr(), metadata.seq_lens_device.data_ptr(), metadata.block_tables.data_ptr())
    first = None
    for lengths in ([129, 145], [127, 131]):
        metadata.seq_lens_device.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        actual = output.cpu().float()
        expected = _reference(
            query, impl.key_cache, impl.value_cache, metadata.block_tables, lengths, query_len, impl.scale
        )
        torch.testing.assert_close(actual, expected, rtol=1.5e-2, atol=1.5e-2)
        assert addresses == (query.data_ptr(), metadata.seq_lens_device.data_ptr(), metadata.block_tables.data_ptr())
        if first is None:
            first = actual.clone()
            query.mul_(0.5)
            impl.value_cache.add_(0.25)
            metadata.block_tables.copy_(torch.tensor([[3, 999], [2, 4]], dtype=torch.int32))
        else:
            assert not torch.allclose(actual, first)
