# SPDX-License-Identifier: Apache-2.0
"""Check union membership against the original per-query sparse selection."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen4_exp.attention_dispatch import build_qsa_request_plan, dispatch_qsa_attention
from vllm_ascend.models.qwen4_exp.gbsa_union import build_qsa_union_inputs, qsa_union_gbsa_attention


@pytest.mark.parametrize("rows,topk", [(1, 8), (7, 8), (8, 8), (9, 8), (17, 512)])
@pytest.mark.parametrize("base_group", [0, (1 << 21) - 1024])
def test_union_preserves_exact_membership(rows, topk, base_group):
    generator = torch.Generator().manual_seed(811)
    indices = torch.randint(0, 1024, (1, rows, topk), generator=generator, dtype=torch.int32) + base_group
    counts = torch.randint(0, topk + 1, (1, rows), generator=generator, dtype=torch.int32)
    visible = torch.arange(rows, dtype=torch.int32) + 4000 + base_group * 4
    # Include empty rows, duplicate selections, padding and partial final groups.
    counts[0, 0] = 0
    for row in range(rows):
        indices[0, row, counts[0, row] :] = -1
    actual = build_qsa_union_inputs(indices.npu(), counts.npu(), visible.npu())
    union_idx, union_count, exact = [tensor.cpu() for tensor in actual]
    for group in range((rows + 7) // 8):
        start, end = group * 8, min(rows, group * 8 + 8)
        selected = [set(indices[0, row, : counts[0, row]].tolist()) for row in range(start, end)]
        expected_union = sorted(set().union(*selected))
        count = int(union_count[0, group])
        assert count == len(expected_union)
        assert union_idx[0, group, :count].tolist() == expected_union
        assert (union_idx[0, group, count:] == -1).all()
        for rank, block in enumerate(expected_union):
            for lane in range(4):
                expected = sum(
                    1 << query_lane
                    for query_lane, blocks in enumerate(selected)
                    if block in blocks and block * 4 + lane < int(visible[start + query_lane])
                )
                assert int(exact[0, group, rank * 4 + lane]) == expected
        assert (exact[0, group, count * 4 :] == 0).all()


def test_large_union_matches_cpu_reference():
    # Exercise the expanded scatter index at the real chunk shape. Keeping that
    # index contiguous is required by the current A3 backend at this size.
    rows, topk = 4096, 512
    indices = torch.empty((1, rows, topk), dtype=torch.int32)
    indices[0, :, :384] = torch.arange(384, dtype=torch.int32)
    indices[0, :, 384:] = (
        512 + torch.arange(rows, dtype=torch.int32).remainder(8)[:, None] * 128 + torch.arange(128, dtype=torch.int32)
    )
    counts = torch.full((1, rows), topk, dtype=torch.int32)
    visible = torch.arange(rows, dtype=torch.int32) + 126977
    expected = build_qsa_union_inputs(indices, counts, visible)
    actual = build_qsa_union_inputs(indices.npu(), counts.npu(), visible.npu())
    for reference, candidate in zip(expected, actual):
        assert torch.equal(reference, candidate.cpu())


@pytest.mark.parametrize("query_lengths", [(1, 7, 8, 9, 15, 16, 17, 19), (512,) * 8])
def test_batched_prefill_dispatch_matches_single_request_launches(query_lengths):
    torch.manual_seed(811)
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    total = starts[-1]
    lengths_cpu = torch.arange(8, dtype=torch.int32) * 5 + 32768
    pages = (int(lengths_cpu.max()) + 127) // 128
    query = torch.randn(total, 3, 256, dtype=torch.bfloat16, device="npu")
    key = torch.randn(8 * pages, 128, 1, 256, dtype=query.dtype, device="npu")
    value = torch.randn_like(key)
    table = torch.randperm(8 * pages, dtype=torch.int32, device="npu").reshape(8, pages)
    lengths = lengths_cpu.npu()
    requests = torch.repeat_interleave(torch.arange(8, dtype=torch.int32), torch.tensor(query_lengths)).npu()
    positions = torch.cat(
        [torch.arange(int(kv) - count, int(kv), dtype=torch.int32) for kv, count in zip(lengths_cpu, query_lengths)]
    ).npu()
    compact = torch.full((total, 515), -1, dtype=torch.int32, device="npu")
    compact[:, :512] = torch.arange(512, dtype=torch.int32, device="npu")
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        rows = slice(start, end)
        qsa_union_gbsa_attention(
            query[rows],
            key,
            value,
            compact[rows],
            table,
            requests[rows],
            positions[rows],
            lengths,
            expected[rows],
            request_index=request,
        )
    plan = build_qsa_request_plan(torch.tensor(starts), lengths_cpu, total, torch.ones(8, dtype=torch.bool))
    assert len(plan.union_batches) == 1
    dispatch_qsa_attention(query, None, None, key, value, compact, table, requests, positions, lengths, actual, plan)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
