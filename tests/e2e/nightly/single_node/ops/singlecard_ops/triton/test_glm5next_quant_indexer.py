# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer-dot KPool cache addressing, quantized scores and causal expansion."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.glm5_next_quant_indexer import _prepare_quant_keys, glm5_next_c8_quant_indexer
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

enable_custom_op()
bootstrap_custom_op_env(include_vendor_lib=True)


@pytest.mark.parametrize("page_rows", [8, 32])
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
    cache = storage[::2] if padded else storage
    table = torch.randperm(blocks, device="npu").int().view(3, width)
    logical = torch.randn(3, width * page_rows, 128, device="npu", dtype=torch.bfloat16)
    k8, ks = torch_npu.npu_dynamic_quant(logical)
    for r in range(3):
        for p in range(width):
            block = table[r, p].item()
            cache[block, :, 0, :128] = k8[r, p * page_rows : (p + 1) * page_rows]
            cache[block, :, 0, 128:] = ks[r, p * page_rows : (p + 1) * page_rows, None].contiguous().view(torch.int8)
    for r, n in enumerate(lengths):
        table[r, (n + page_rows - 1) // page_rows :] = 2**30
    lens = torch.tensor(lengths, device="npu", dtype=torch.int64)
    prepared = _prepare_quant_keys(cache, table, lens, 600)
    keys, scales, packed_table = prepared
    for r, n in enumerate(lengths):
        torch.testing.assert_close(keys.reshape(3, -1, 128)[r, :n], k8[r, :n], rtol=0, atol=0)
        torch.testing.assert_close(scales.reshape(3, -1)[r, :n], ks[r, :n].half(), rtol=0, atol=0)
        assert (keys.reshape(3, -1, 128)[r, n:] == 0).all()
        assert (scales.reshape(3, -1)[r, n:] == 0).all()
    ends = torch.tensor([5, 8, 9], device="npu", dtype=torch.int64)
    positions = torch.tensor([2397, 2398, 2399, 2400, 2401, 8, 9, 10, 0], device="npu", dtype=torch.int64)
    tokens = 9 + padded_queries
    if padded_queries:
        positions = torch.cat((positions, torch.zeros(padded_queries, device="npu", dtype=torch.int64)))
    q = torch.randn(tokens, 32, 128, device="npu", dtype=torch.bfloat16)
    w = torch.randn(tokens, 32, device="npu", dtype=torch.bfloat16)
    meta = SimpleNamespace(block_table=table, seq_lens=lens, c8_quant_workspace=None, c8_quant_metadata=None)
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


def test_integer_kpool_workspace_budget():
    cache = torch.empty(1, 32, 1, 132, dtype=torch.int8)
    table = torch.empty(128, 4096, dtype=torch.int32)
    lengths = torch.empty(128, dtype=torch.int64)
    assert _prepare_quant_keys(cache, table, lengths, 131072) is None


@pytest.mark.parametrize("pool_size,topk", [(0, 2048), (3, 2046), (256, 2048), (4, 8196)])
def test_integer_kpool_operator_limits(pool_size, topk):
    query = torch.empty(1, 32, 128)
    unused = SimpleNamespace()
    assert glm5_next_c8_quant_indexer(query, query, query, unused, unused, 0, topk, pool_size, None) is None
