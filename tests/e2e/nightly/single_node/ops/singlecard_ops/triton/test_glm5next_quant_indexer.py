# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer-dot KPool cache addressing, quantized scores and causal expansion."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.glm5_next_quant_indexer import glm5_next_c8_quant_indexer, view_quant_kpool_cache
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

enable_custom_op()
bootstrap_custom_op_env(include_vendor_lib=True)


@pytest.mark.parametrize("page_rows", [16, 32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("padded_queries", [0, 2, 503])
def test_integer_kpool_cache_and_causal_expansion(page_rows, padded, padded_queries):
    torch.manual_seed(18534)
    lengths = [600, 2, 0]
    qlens = [5, 3, 1]
    tails = [2, 3, 1]
    width = (600 + page_rows - 1) // page_rows
    blocks = 3 * width
    storage = torch.zeros(blocks * (2 if padded else 1), page_rows, 1, 132, device="npu", dtype=torch.int8)
    cache = storage[1::2] if padded else storage
    keys, scales = view_quant_kpool_cache(cache)
    table = torch.randperm(blocks, device="npu").int().view(3, width)
    logical = torch.randn(3, width * page_rows, 128, device="npu", dtype=torch.bfloat16)
    k8, ks = torch_npu.npu_dynamic_quant(logical)
    for r in range(3):
        for p in range(width):
            block = table[r, p].item()
            keys[block, :, 0, :] = k8[r, p * page_rows : (p + 1) * page_rows]
            scales[block, :, 0] = ks[r, p * page_rows : (p + 1) * page_rows].half()
    for r, n in enumerate(lengths):
        table[r, (n + page_rows - 1) // page_rows :] = 2**30
    lens = torch.tensor(lengths, device="npu", dtype=torch.int64)
    ends = torch.tensor([5, 8, 9], device="npu", dtype=torch.int64)
    positions = torch.tensor([2397, 2398, 2399, 2400, 2401, 8, 9, 10, 0], device="npu", dtype=torch.int64)
    tokens = 9 + padded_queries
    if padded_queries:
        positions = torch.cat((positions, torch.zeros(padded_queries, device="npu", dtype=torch.int64)))
    q = torch.randn(tokens, 32, 128, device="npu", dtype=torch.bfloat16)
    w = torch.randn(tokens, 32, device="npu", dtype=torch.bfloat16)
    meta = SimpleNamespace(block_table=table, seq_lens=lens, c8_quant_metadata=None)
    qm = SimpleNamespace(
        cum_query_lens=ends,
        seq_lens=lens,
        pool_tail=torch.tensor(tails, device="npu", dtype=torch.int64),
        positions=positions,
    )
    output = torch.full((tokens + 1, 2064), -777, device="npu", dtype=torch.int32)
    result = glm5_next_c8_quant_indexer(q, cache, w, meta, qm, 600, 2048, 4, output)
    assert result.shape == (tokens, 1, 2051)
    assert (result[9:] == -1).all()
    result = result.cpu().squeeze(1)
    qi, qs = torch_npu.npu_dynamic_quant(q)
    qi, qs = qi.cpu().float(), qs.cpu().half().float()
    kic, ksp = k8.cpu().float(), ks.cpu().half().float()
    weights = w.cpu().half().float()
    row = 0
    for req, n in enumerate(qlens):
        for _ in range(n):
            position = int(positions[row].item())
            visible = (position + 1) // 4
            count = min(visible, 512)
            dots = (qi[row] @ kic[req].T / 1024).clamp_min(0).half().float()
            scores = (dots * (weights[row] * qs[row]).half().float()[:, None]).sum(0) * ksp[req]
            gold = torch.topk(scores[:visible], count).indices.tolist()
            expected = set(v * 4 + i for v in gold for i in range(4))
            expected.update(range(visible * 4, position + 1))
            actual = [x for x in result[row].tolist() if x >= 0]
            recall = len(set(actual) & expected) / max(1, len(expected))
            assert recall >= 0.98, (row, recall)
            assert len(actual) == len(set(actual)) == len(expected), (row, len(actual), len(expected))
            assert all(x <= position for x in actual), (row, position)
            row += 1
    assert (output[:tokens, 2051:] == -1).all()
    assert (output[tokens:] == -777).all()
    repeated = glm5_next_c8_quant_indexer(q, cache, w, meta, qm, 600, 2048, 4, output).cpu().squeeze(1)
    torch.testing.assert_close(repeated, result, rtol=0, atol=0)


def test_integer_kpool_views_share_padded_offset_storage():
    storage = torch.empty(5, 32, 1, 132, dtype=torch.int8)
    cache = storage[1::2]
    keys, scales = view_quant_kpool_cache(cache)
    assert keys.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    assert scales.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    keys.fill_(7)
    scales.fill_(0.5)
    assert (keys == 7).all()
    assert (scales == 0.5).all()
    assert keys.stride(0) == cache.stride(0)
    assert scales.storage_offset() == (cache.storage_offset() + 32 * 128) // 2


@pytest.mark.parametrize("pool_size,topk", [(0, 2048), (3, 2046), (256, 2048), (4, 8196)])
def test_integer_kpool_operator_limits(pool_size, topk):
    query = torch.empty(1, 32, 128)
    unused = SimpleNamespace()
    with pytest.raises(ValueError, match="power-of-two"):
        glm5_next_c8_quant_indexer(query, query, query, unused, unused, 0, topk, pool_size, None)


@pytest.mark.parametrize("tokens", [1, 6, 16])
def test_integer_kpool_graph_replay_updates_lengths_and_pages(tokens):
    torch.manual_seed(18535)
    cache = torch.zeros(32, 32, 1, 132, device="npu", dtype=torch.int8)
    keys, scales = view_quant_kpool_cache(cache)
    keys.copy_(torch.randint(-100, 100, keys.shape, device="npu", dtype=torch.int8))
    scales.fill_(0.01)
    table = torch.arange(32, device="npu", dtype=torch.int32).view(1, 32)
    lens = torch.zeros(1, device="npu", dtype=torch.int64)
    ends = torch.zeros_like(lens)
    tails = torch.zeros_like(lens)
    positions = torch.zeros(tokens, device="npu", dtype=torch.int64)
    q = torch.randn(tokens, 32, 128, device="npu", dtype=torch.bfloat16)
    weights = torch.randn(tokens, 32, device="npu", dtype=torch.bfloat16)
    qm = SimpleNamespace(cum_query_lens=ends, seq_lens=lens, pool_tail=tails, positions=positions)

    def run(output):
        meta = SimpleNamespace(block_table=table, c8_quant_metadata=None)
        return glm5_next_c8_quant_indexer(q, cache, weights, meta, qm, 1024, 2048, 4, output)

    output = torch.empty(tokens, 2064, device="npu", dtype=torch.int32)
    run(output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run(output)
    for length, live in [(0, 1), (10, tokens), (601, tokens), (1000, max(1, tokens - 2))]:
        lens.fill_(length)
        tails.fill_(3)
        ends.fill_(live)
        positions[:live].copy_(torch.arange(length * 4 + 3 - live, length * 4 + 3, device="npu"))
        table.copy_(table.flip(1))
        q.normal_()
        expected = run(None).clone()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_integer_kpool_eight_one_million_token_histories():
    requests, pools, page_rows = 8, 262144, 32
    pages = requests * pools // page_rows
    cache = torch.zeros(pages, page_rows, 1, 132, device="npu", dtype=torch.int8)
    keys, scales = view_quant_kpool_cache(cache)
    keys.fill_(1)
    scales.fill_(0.01)
    table = torch.arange(pages, device="npu", dtype=torch.int32).view(requests, -1)
    lens = torch.full((requests,), pools, device="npu", dtype=torch.int64)
    qm = SimpleNamespace(
        cum_query_lens=torch.arange(1, requests + 1, device="npu", dtype=torch.int64),
        seq_lens=lens,
        pool_tail=torch.zeros(requests, device="npu", dtype=torch.int64),
        positions=torch.full((requests,), 1048575, device="npu", dtype=torch.int64),
    )
    meta = SimpleNamespace(block_table=table, c8_quant_metadata=None)
    query = torch.ones(requests, 32, 128, device="npu", dtype=torch.bfloat16)
    weights = torch.ones(requests, 32, device="npu", dtype=torch.bfloat16)
    result = glm5_next_c8_quant_indexer(query, cache, weights, meta, qm, pools, 2048, 4, None).cpu()
    for row in result[:, 0]:
        assert (row[:2048] >= 0).all() and (row[:2048] < 1048576).all()
        assert row[:2048].unique().numel() == 2048
        assert (row[2048:] == -1).all()


def test_integer_kpool_missing_operator_is_an_error(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace())
    query = torch.empty(1, 32, 128)
    unused = SimpleNamespace()
    with pytest.raises(RuntimeError, match="requires the QLI v2 integer operators"):
        glm5_next_c8_quant_indexer(query, query, query, unused, unused, 0, 2048, 4, None)
