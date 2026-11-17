# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single A3 device tests; no model weights or distributed service required."""

from dataclasses import replace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.flash_attn_npu import FlashAttnNPUConfig, FlashAttnNPUImpl
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


@pytest.fixture
def case():
    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 MLA operator integration")
    pytest.importorskip("flash_attn_npu_3")
    torch.manual_seed(23)

    def make(dtype, heads, causal):
        # Includes multi-token queries, an empty-KV row, an empty-query row,
        # trailing graph padding and nontrivial physical page indices.
        q = (torch.randn(7, heads, 576) * 0.1).to(dtype).npu()
        caches = []
        for dim in (512, 64):
            stride = 128 * dim + 32
            raw = (torch.randn(6 * stride + 16) * 0.1).to(dtype).npu()
            caches.append(raw.as_strided((6, 128, 1, dim), (stride, dim, dim, 1), 16))
        lengths = torch.tensor([132, 2, 0, 0, 0], dtype=torch.int32, device="npu")
        cu = torch.tensor([0, 3, 4, 5, 5, 7], dtype=torch.int32, device="npu")
        table = torch.tensor([[3, 1], [4, 2], [0, 0], [0, 0], [0, 0]], dtype=torch.int32, device="npu")
        adapter = replace(
            FlashAttnNPUImpl.load(FlashAttnNPUConfig(heads, 0.125, 3 if causal else 0, True), dtype),
            max_seqlen_q=q.shape[0],
            max_seqlen_k=table.shape[1] * 128,
        )
        return adapter, q, tuple(caches), lengths, cu, table

    return make


def reference(q, caches, lengths, cu, table, causal):
    q, lengths, cu, table = (tensor.cpu() for tensor in (q, lengths, cu, table))
    caches = tuple(cache.cpu() for cache in caches)
    out = torch.zeros(q.shape[1], q.shape[0], 512, dtype=torch.float64)
    lse = torch.full(q.shape[:2][::-1], -torch.inf, dtype=torch.float64)
    for row, length in enumerate(lengths.tolist()):
        if not length:
            continue
        indices = torch.arange(length)
        pages = table[row, indices // 128].long()
        keys = torch.cat([cache[pages, indices % 128, 0] for cache in caches], -1).double()
        start, end = cu[row : row + 2].tolist()
        for token in range(start, end):
            visible = length - (end - token) + 1 if causal else length
            scores = q[token].double() @ keys[:visible].T * 0.125
            out[:, token] = scores.softmax(-1) @ keys[:visible, :512]
            lse[:, token] = scores.logsumexp(-1)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [8, 32, 96])
@pytest.mark.parametrize("causal", [False, True])
def test_a3_mla_paged_attention(case, dtype, heads, causal):
    adapter, q, caches, lengths, cu, table = case(dtype, heads, causal)
    before = tuple(cache.clone() for cache in caches)
    metadata = adapter.build_metadata(lengths, cu)
    out, lse = adapter.attention(
        q,
        caches,
        block_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        metadata=metadata,
    )
    expected, expected_lse = reference(q, caches, lengths, cu, table, causal)
    torch.testing.assert_close(out.cpu().double(), expected, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(lse[:, :4].cpu().double(), expected_lse[:, :4], atol=3e-3, rtol=3e-3)
    assert not torch.isfinite(lse[:, 4:]).any()
    for cache, snapshot in zip(caches, before):
        torch.testing.assert_close(cache, snapshot)


@pytest.mark.parametrize("causal", [False, True])
def test_a3_mla_metadata_and_attention_graph_replay(case, causal):
    adapter, q, caches, lengths, cu, table = case(torch.bfloat16, 8, causal)
    schedule = adapter.build_metadata(lengths, cu)
    kwargs = dict(block_table=table, cache_seqlens=lengths, cu_seqlens_q=cu, metadata=schedule)

    def forward():
        schedule.copy_(adapter.build_metadata(lengths, cu))
        return adapter.attention(q, caches, **kwargs)

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, lse = forward()
    address = schedule.data_ptr()
    for next_length in (129, 131, 132):
        lengths[0].fill_(next_length)
        graph.replay()
        torch.npu.synchronize()
        expected, expected_lse = reference(q, caches, lengths, cu, table, causal)
        torch.testing.assert_close(output.cpu().double(), expected, atol=3e-3, rtol=2e-2)
        torch.testing.assert_close(lse[:, :4].cpu().double(), expected_lse[:, :4], atol=3e-3, rtol=3e-3)
        assert schedule.data_ptr() == address
