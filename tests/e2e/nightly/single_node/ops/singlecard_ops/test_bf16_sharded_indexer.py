# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.bf16_sharded_indexer import (
    bf16_paged_indexer_candidates,
)
from vllm_ascend.ops.lightning_indexer_v2 import (
    build_lightning_indexer_v2_plan,
    has_lightning_indexer_v2,
    lightning_indexer_v2_candidates,
)


@torch.inference_mode()
def test_bf16_sharded_indexer_npu_matches_cpu():
    torch_npu.npu.set_device(0)
    generator = torch.Generator().manual_seed(37)
    query = torch.randn((4, 32, 128), generator=generator).bfloat16()
    query[1].zero_()
    weights = torch.randn((4, 32), generator=generator).bfloat16()
    keys = torch.randn((24, 128, 1, 128), generator=generator).bfloat16()
    table = torch.tensor([[*range(2, 20)], [*range(23, 5, -1)]], dtype=torch.int32)
    ends = torch.tensor([2, 3], dtype=torch.int32)
    visible = torch.tensor([127, 2049, 2101, 0], dtype=torch.int32)
    inputs = (query, weights, keys, table, ends, visible)
    expected_ids, expected_scores = bf16_paged_indexer_candidates(*inputs, 2048, first_valid_page=2)
    device_inputs = tuple(tensor.npu() for tensor in inputs)
    ids, scores = bf16_paged_indexer_candidates(*device_inputs, 2048, first_valid_page=2)
    ids, scores = ids.cpu(), scores.cpu()
    # Different FP32 reduction order can reorder nearly tied stochastic rows.
    # Demand exact IDs for deterministic ties and padding, and compare the
    # sorted score distributions plus ID set where the cutoff is separated.
    torch.testing.assert_close(ids[[1, 3]], expected_ids[[1, 3]])
    torch.testing.assert_close(ids[0].sort().values, expected_ids[0].sort().values)
    torch.testing.assert_close(scores, expected_scores, atol=5e-3, rtol=5e-4)


@torch.inference_mode()
def test_bf16_sharded_indexer_graph_replay_reads_new_pages_and_visible_lengths():
    torch_npu.npu.set_device(0)
    # Match the production table width. Most scores are tied, and the final
    # page outranks the earlier ones for row 1 after the length grows.
    query = torch.zeros((4, 32, 128), dtype=torch.bfloat16, device="npu")
    query[1].fill_(1)
    weights = torch.ones((4, 32), dtype=query.dtype, device=query.device)
    page_size = 896
    num_pages = 19  # Rank-local table capacity at 131072 tokens and DCP8.
    keys = torch.zeros((num_pages + 1, page_size, 1, 128), dtype=query.dtype, device=query.device)
    keys[-1].fill_(1)
    table = torch.zeros((2, num_pages), dtype=torch.int32, device=query.device)
    ends = torch.tensor([4, 4], dtype=torch.int32, device=query.device)
    visible = torch.zeros(4, dtype=torch.int32, device=query.device)
    request_rows = torch.zeros_like(visible)
    candidate_inputs = (query, weights, keys, table, ends, visible)
    topk = 2048
    bf16_paged_indexer_candidates(*candidate_inputs, topk, full_graph=True, request_rows=request_rows)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        ids, scores = bf16_paged_indexer_candidates(*candidate_inputs, topk, full_graph=True, request_rows=request_rows)

    table[0].copy_(torch.arange(1, num_pages + 1, dtype=torch.int32))
    table[1, 0] = num_pages
    ends.copy_(torch.tensor([2, 3], dtype=torch.int32))
    request_rows.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    visible.copy_(torch.tensor([page_size - 1, num_pages * page_size, 5, 0], dtype=torch.int32))
    graph.replay()
    expected_ids, expected_scores = bf16_paged_indexer_candidates(*candidate_inputs, topk)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu())
    torch.testing.assert_close(scores.cpu(), expected_scores.cpu())

    visible.zero_()
    graph.replay()
    torch.testing.assert_close(ids.cpu(), torch.full_like(expected_ids.cpu(), -1))
    torch.testing.assert_close(scores.cpu(), torch.full_like(expected_scores.cpu(), -torch.inf))


def _reference_candidates(query, weights, keys, table, request_rows, visible, topk, first_valid_page):
    expected_ids = torch.full((len(query), 1, topk), -1, dtype=torch.int32)
    expected_scores = torch.full(expected_ids.shape, -torch.inf, dtype=torch.float32)
    for row, length in enumerate(visible.tolist()):
        ids = torch.arange(length)
        page_size = keys.shape[1]
        pages = table[request_rows[row], ids // page_size].long()
        valid = (pages >= first_valid_page) & (pages < len(keys))
        ids = ids[valid]
        row_keys = keys[pages[valid], ids % page_size, 0].float()
        scores = (row_keys @ query[row].float().T).relu() @ weights[row].float()
        order = scores.argsort(descending=True, stable=True)[:topk]
        expected_ids[row, 0, : len(order)] = ids[order].int()
        expected_scores[row, 0, : len(order)] = scores[order]
    return expected_ids, expected_scores


@torch.inference_mode()
def test_native_bf16_indexer_dcp_windows_and_graph_replay(require_native_indexer):
    rows, heads, dim, page_size, topk = 6, 32, 128, 128, 2048
    first_valid_page = 2
    generator = torch.Generator().manual_seed(802)
    # Integer-valued inputs produce exact FP32 scores even with different
    # reduction orders. Negative weights exercise signed candidate scores.
    query = torch.randint(-1, 2, (rows, heads, dim), generator=generator).bfloat16()
    query[1].zero_()  # More than K identical scores across two windows.
    weights = torch.randint(-2, 3, (rows, heads), generator=generator).float()
    keys = torch.randint(-2, 3, (22, page_size, 1, dim), generator=generator).bfloat16()
    keys[:first_valid_page].fill_(1000)
    # One high-scoring token lives in the second window, past the cutoff.
    keys[18, 123].fill_(64)
    query[0].fill_(1)
    weights[0].fill_(1)
    table = torch.arange(2, len(keys), dtype=torch.int32).repeat(2, 1)
    table[1] = table[1].flip(0)
    table[1, 0] = first_valid_page - 1
    table[1, 4] = len(keys)
    request_rows = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.int32)
    device_q, device_w, device_k, device_table, device_rows = (
        value.npu() for value in (query, weights, keys, table, request_rows)
    )
    device_visible = torch.zeros(rows, device="npu", dtype=torch.int32)

    def run():
        plan = build_lightning_indexer_v2_plan(
            device_table, device_rows, device_visible, len(keys), page_size, first_valid_page, heads
        )
        return lightning_indexer_v2_candidates(device_q, device_w, device_k, plan, topk)

    stripe = 768
    cycle = 8 * stripe

    def visible_lengths(rank):
        global_lengths = [
            3 * cycle + 128,
            3 * cycle,
            (16 + rank) * stripe + 513,
            rank * stripe + 127,
            rank * stripe + 129,
            0,
        ]
        # Count ownership directly, independently of the DCP metadata formula.
        return torch.tensor(
            [sum(token // stripe % 8 == rank for token in range(n)) for n in global_lengths], dtype=torch.int32
        )

    def check(actual, visible):
        expected = _reference_candidates(query, weights, keys, table, request_rows, visible, topk, first_valid_page)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), reference, atol=0, rtol=0)

    visible = visible_lengths(0)
    device_visible.copy_(visible)
    check(run(), visible)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = run()

    # The same graph sees a different shard tail, request mapping and physical
    # page order. Row 1 still crosses the top-K boundary with identical scores.
    visible = visible_lengths(7)
    table[0] = table[0].flip(0)
    request_rows[2] = 1
    device_table.copy_(table)
    device_rows.copy_(request_rows)
    device_visible.copy_(visible)
    graph.replay()
    check(captured, visible)

    device_visible.zero_()
    graph.replay()
    check(captured, torch.zeros_like(visible))


@pytest.fixture
def require_native_indexer():
    if not is_950():
        pytest.skip("requires the A5 BF16 indexer path")
    torch_npu.npu.set_device(0)
    if not has_lightning_indexer_v2():
        pytest.skip("requires the BF16 LightningIndexerV2 and metadata operators")


@torch.inference_mode()
def test_native_bf16_indexer_causal_prefill(require_native_indexer):
    rows, heads, page_size, topk = 4, 32, 128, 2048
    query = torch.ones(rows, heads, 128, dtype=torch.bfloat16)
    weights = torch.ones(rows, heads)
    keys = torch.ones(18, page_size, 1, 128, dtype=torch.bfloat16)
    keys[1, 0].zero_()  # Unique losing token when the second query sees K+1 keys.
    pages = torch.arange(1, len(keys), dtype=torch.int32)
    table = torch.stack([pages, pages.flip(0)])
    cu_query = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    kv_lengths = torch.tensor([2049, 127], dtype=torch.int32, device="npu")
    ops = torch.ops._C_ascend
    metadata = ops.npu_lightning_indexer_v2_metadata(cu_query, kv_lengths, heads, topk, True)
    ids, values = ops.npu_lightning_indexer_v2(
        query.npu(),
        keys.npu(),
        weights.npu(),
        cu_query,
        kv_lengths,
        table.npu(),
        metadata,
        topk,
        causal=True,
        return_values=False,
    )

    expected_ids, _ = _reference_candidates(
        query, weights, keys, table, torch.tensor([0, 0, 1, 1]), torch.tensor([2048, 2049, 126, 127]), topk, 1
    )
    # Ordinary SFA consumes the selected IDs, including the native short-row
    # ordering; this fixture has an unambiguous set at each row's cutoff.
    torch.testing.assert_close(ids.cpu().sort().values, expected_ids.sort().values)
    assert values.numel() == 0
