# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("top_k", [3, 4])
def test_route_transport_preserves_native_bits(monkeypatch, weight_dtype, id_dtype, top_k):
    layer = object.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=1, pcp_size=2, is_sequence_parallel=False)
    weights = torch.tensor([[0.0, -0.0, float("inf"), float("nan")]], dtype=weight_dtype)[:, :top_k]
    ids = torch.tensor([[0, 127, 255, 287]], dtype=id_dtype)[:, :top_k]
    group = SimpleNamespace(all_gather=lambda x, dim: torch.cat((x, x), dim))
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_pcp=1))
    _, actual_w, actual_i = layer.prepare_with_local_topk(
        torch.ones(1, 16), torch.ones(1, 288), lambda h, r: (weights, ids)
    )
    assert actual_w.dtype == weight_dtype and actual_i.dtype == id_dtype
    assert torch.equal(actual_w.view(torch.uint8), torch.cat([weights] * 2).view(torch.uint8))
    assert torch.equal(actual_i, torch.cat([ids] * 2))


@pytest.mark.parametrize("d,p", [(d, p) for d in range(2) for p in range(3)])
def test_dp_then_pcp_padding_and_finalize_with_distinct_empty_shards(monkeypatch, d, p):
    # DP ranks own different requests; PCP ranks own different token fragments.
    sizes = ((2, 1, 0), (0, 3, 1))
    torch.manual_seed(61)
    hidden = [[torch.randn(n, 16) for n in row] for row in sizes]
    logits = [[torch.randn(n, 12) for n in row] for row in sizes]
    dp_max = [max(sizes[d][p] for d in range(2)) for p in range(3)]
    dp_hidden = [
        torch.cat([F.pad(hidden[d][p], (0, 0, 0, dp_max[p] - sizes[d][p])) for d in range(2)]) for p in range(3)
    ]
    dp_logits = [
        torch.cat([F.pad(logits[d][p], (0, 0, 0, dp_max[p] - sizes[d][p])) for d in range(2)]) for p in range(3)
    ]
    pcp_max = max(map(len, dp_hidden))
    padded_hidden = [F.pad(x, (0, 0, 0, pcp_max - len(x))) for x in dp_hidden]
    padded_logits = [F.pad(x, (0, 0, 0, pcp_max - len(x))) for x in dp_logits]
    global_hidden, global_logits = torch.cat(padded_hidden), torch.cat(padded_logits)

    def route(h, r):
        scores = r.abs() + 1
        ids = scores.topk(3, dim=-1).indices.int()
        weights = scores.gather(1, ids.long())
        return weights / weights.sum(-1, keepdim=True), ids

    expected_w, expected_i = route(global_hidden, global_logits)
    packed = torch.cat((expected_w.view(torch.int32), expected_i), dim=-1)
    dp_calls, pcp_calls = [], []

    def dp_gather(x, dim):
        dp_calls.append(x)
        expected = hidden[d][p] if len(dp_calls) == 1 else logits[d][p]
        torch.testing.assert_close(x, F.pad(expected, (0, 0, 0, dp_max[p] - len(expected))))
        return dp_hidden[p] if len(dp_calls) == 1 else dp_logits[p]

    def pcp_gather(x, dim):
        pcp_calls.append(x)
        if len(pcp_calls) == 1:
            assert torch.equal(x, padded_hidden[p])
            return global_hidden
        assert torch.equal(x, packed[p * pcp_max : (p + 1) * pcp_max])
        return packed

    dp = SimpleNamespace(all_gather=dp_gather, reduce_scatter=lambda x, dim: x.chunk(2, dim=dim)[d].contiguous())
    pcp = SimpleNamespace(all_gather=pcp_gather, reduce_scatter=lambda x, dim: x.chunk(3, dim=dim)[p].contiguous())
    layer = object.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(dp_size=2, dp_group=dp, pcp_size=3, is_sequence_parallel=False)
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: pcp)
    monkeypatch.setattr(prepare_module, "get_dp_group", lambda: dp)
    monkeypatch.setattr(
        prepare_module,
        "_EXTRA_CTX",
        SimpleNamespace(max_tokens_across_dp=dp_max[p], max_tokens_across_pcp=pcp_max),
    )
    actual_h, actual_w, actual_i = layer.prepare_with_local_topk(hidden[d][p], logits[d][p], route)
    assert torch.equal(actual_h, global_hidden)
    torch.testing.assert_close(actual_w, expected_w, rtol=0, atol=0)
    assert torch.equal(actual_i, expected_i)
    assert layer.num_tokens == sizes[d][p]
    assert layer.num_tokens_pcp == len(dp_hidden[p])
    assert len(dp_calls) == len(pcp_calls) == 2
    assert torch.equal(layer.finalize(actual_h, reduce_results=False), hidden[d][p])
