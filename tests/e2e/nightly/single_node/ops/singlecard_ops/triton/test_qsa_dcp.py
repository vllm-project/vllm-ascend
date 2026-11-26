# SPDX-License-Identifier: Apache-2.0
"""Single-NPU numerical gate for QSA local KV shards and GBSA LSE output."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen4_exp.gbsa import qsa_gbsa_local_attention
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs


@pytest.mark.parametrize("world,interleave,length", [(2, 4, 9), (4, 16, 3), (4, 128, 2051), (2, 4, 4102)])
def test_qsa_dcp_partial_attention(world, interleave, length):
    torch.manual_seed(19)
    device = "npu"
    query = torch.randn(1, 4, 256, dtype=torch.bfloat16, device=device)
    keys = torch.randn(length, 1, 256, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    groups = min(512, length // 4)
    selected = torch.randperm(length // 4, device=device)[:groups]
    lengths = torch.tensor([length], dtype=torch.int32, device=device)
    requests = torch.zeros(1, dtype=torch.int32, device=device)

    def run(owned, size, rank):
        count = owned.numel()
        pages = max(1, (count + 127) // 128)
        table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device=device).view(1, -1)
        k = torch.zeros(pages * 2, 128, 1, 256, dtype=torch.bfloat16, device=device)[::2]
        v = torch.zeros_like(k)
        local = torch.arange(count, device=device)
        page_ids = table[0, local // 128].long()
        k[page_ids, local % 128] = keys[owned]
        v[page_ids, local % 128] = values[owned]
        stripe = interleave // 4
        winners = selected[selected // stripe % size == rank]
        local_indices = torch.full((1, 515), -1, dtype=torch.int32, device=device)
        local_indices[0, : winners.numel()] = winners // (size * stripe) * stripe + winners % stripe
        local_indices[0, 512 : 512 + length % 4] = torch.arange(length // 4 * 4, length, device=device)
        return qsa_gbsa_local_attention(
            query,
            k,
            v,
            local_indices,
            table,
            requests,
            lengths - 1,
            lengths,
            dcp_size=size,
            dcp_rank=rank,
            dcp_interleave=interleave,
        )

    tokens = torch.arange(length, device=device)
    selected_tokens = torch.cat(
        ((selected[:, None] * 4 + torch.arange(4, device=device)).flatten(), tokens[length // 4 * 4 :])
    )
    logits = torch.einsum("hd,sd->hs", query[0].float(), keys[selected_tokens, 0].float()) * query.shape[-1] ** -0.5
    expected = (logits.softmax(-1) @ values[selected_tokens, 0].float()).unsqueeze(0)
    partials, lses = [], []
    for rank in range(world):
        partial, lse = run(tokens[tokens // interleave % world == rank], world, rank)
        partials.append(partial.float())
        lses.append(lse)
    lse = torch.stack(lses)
    actual = (torch.stack(partials) * (lse - lse.logsumexp(0)).exp()).sum(0)
    torch.testing.assert_close(actual, expected.float(), rtol=0.02, atol=0.02)


@pytest.mark.parametrize("rank,owned", [(0, 1026), (1, 1024)])
def test_frozen_tail_on_dcp_graph_replay(rank, owned):
    compact = torch.full((3, 515), -1, dtype=torch.int32, device="npu")
    compact[:, :256] = torch.arange(256, device="npu")
    compact[:, 512:514] = torch.tensor([2048, 2049], device="npu")
    positions = torch.full((3,), 2049, dtype=torch.int32, device="npu")
    lengths = torch.tensor([2052], dtype=torch.int32, device="npu")
    requests = torch.zeros_like(positions)
    table = torch.zeros((1, 9), dtype=torch.int32, device="npu")

    def prepare():
        return prepare_qsa_gbsa_inputs(
            compact, positions, lengths, requests, table, 1, dcp_size=2, dcp_rank=rank, dcp_interleave=4
        )

    prepare()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _, counts, visible, _, _ = prepare()
    # The step-0 buffer is still live when a draft row becomes graph padding.
    positions.copy_(torch.tensor([2050, -1, 2051], dtype=torch.int32, device="npu"))
    graph.replay()
    assert visible.tolist() == [owned, 0, owned]
    assert counts.tolist() == [[256, 0, 256]]
    graph.reset()
