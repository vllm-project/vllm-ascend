# SPDX-License-Identifier: Apache-2.0
"""Check union membership against the original per-query sparse selection."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen4_exp.attention_dispatch import (
    build_qsa_request_plan,
    build_qsa_union_group_row_map,
    dispatch_qsa_attention,
    prepare_qsa_union_batches,
)
from vllm_ascend.models.qwen4_exp.gbsa_union import (
    _build_qsa_union_inputs_torch,
    build_qsa_union_inputs_from_compact,
    qsa_union_gbsa_attention,
)


def _reference_compact(compact, visible, row_map):
    input_groups = compact.shape[1] - 3
    topk = input_groups + int(input_groups < 512)
    groups = row_map.shape[0]
    sparse = torch.full((1, groups * 8, topk), -1, dtype=torch.int32)
    counts = torch.zeros((1, groups * 8), dtype=torch.int32)
    padded_visible = torch.zeros(groups * 8, dtype=torch.int32)
    for group, rows in enumerate(row_map.tolist()):
        for lane, row in enumerate(rows):
            if row < 0:
                continue
            seen = int(visible[row])
            selected = compact[row, :input_groups]
            selected = selected[(selected >= 0) & (selected < seen // 4)]
            has_tail = int(seen % 4 != 0)
            selected = selected[: topk - has_tail]
            if has_tail:
                selected = torch.cat((selected, torch.tensor([seen // 4], dtype=torch.int32)))
            destination = group * 8 + lane
            sparse[0, destination, : selected.numel()] = selected
            counts[0, destination] = selected.numel()
            padded_visible[destination] = seen
    return _build_qsa_union_inputs_torch(sparse, counts, padded_visible)


@pytest.mark.parametrize("rows,input_groups", [(1, 7), (7, 7), (8, 7), (9, 7), (17, 512)])
@pytest.mark.parametrize("base_group", [0, (1 << 21) - 1024])
def test_union_preserves_exact_membership(rows, input_groups, base_group):
    generator = torch.Generator().manual_seed(811)
    compact = torch.randint(0, 1024, (rows, input_groups + 3), generator=generator, dtype=torch.int32) + base_group
    compact[0, :input_groups] = -1
    visible = torch.arange(rows, dtype=torch.int32) + 4000 + base_group * 4
    row_map = torch.tensor(build_qsa_union_group_row_map((0, rows)), dtype=torch.int32)
    expected = _reference_compact(compact, visible, row_map)
    actual = build_qsa_union_inputs_from_compact(compact.npu(), visible.npu(), row_map.npu())
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(reference, candidate.cpu(), rtol=0, atol=0)


def test_large_union_matches_cpu_reference():
    rows = 4096
    compact = torch.full((rows, 515), -1, dtype=torch.int32)
    compact[:, :384] = torch.arange(384, dtype=torch.int32)
    compact[:, 384:512] = (
        512 + torch.arange(rows, dtype=torch.int32).remainder(8)[:, None] * 128 + torch.arange(128, dtype=torch.int32)
    )
    visible = torch.arange(rows, dtype=torch.int32) + 126977
    row_map = torch.tensor(build_qsa_union_group_row_map((0, rows)), dtype=torch.int32)
    expected = _reference_compact(compact, visible, row_map)
    actual = build_qsa_union_inputs_from_compact(compact.npu(), visible.npu(), row_map.npu())
    for reference, candidate in zip(expected, actual):
        torch.testing.assert_close(reference, candidate.cpu(), rtol=0, atol=0)


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
        single_plan = build_qsa_request_plan(
            torch.tensor([0, end - start]),
            lengths_cpu[request : request + 1],
            end - start,
            torch.tensor([True]),
        )
        single_plan = prepare_qsa_union_batches(
            single_plan,
            positions[rows] + 1,
            lengths[request : request + 1],
            table[request : request + 1],
        )
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
            prepared=single_plan.prepared_union_batches[0],
        )
    plan = build_qsa_request_plan(torch.tensor(starts), lengths_cpu, total, torch.ones(8, dtype=torch.bool))
    plan = prepare_qsa_union_batches(plan, positions + 1, lengths, table)
    assert len(plan.union_batches) == 1
    dispatch_qsa_attention(query, None, None, key, value, compact, table, requests, positions, lengths, actual, plan)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
