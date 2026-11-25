# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu

from vllm_ascend.ops.bf16_sharded_indexer import bf16_paged_indexer_candidates


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
    candidate_inputs = (query, weights, keys, table, ends, visible)
    topk = 2048
    bf16_paged_indexer_candidates(*candidate_inputs, topk, full_graph=True)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        ids, scores = bf16_paged_indexer_candidates(*candidate_inputs, topk, full_graph=True)

    table[0].copy_(torch.arange(1, num_pages + 1, dtype=torch.int32))
    table[1, 0] = num_pages
    ends.copy_(torch.tensor([2, 3], dtype=torch.int32))
    visible.copy_(torch.tensor([page_size - 1, num_pages * page_size, 5, 0], dtype=torch.int32))
    graph.replay()
    expected_ids, expected_scores = bf16_paged_indexer_candidates(*candidate_inputs, topk)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu())
    torch.testing.assert_close(scores.cpu(), expected_scores.cpu())

    visible.zero_()
    graph.replay()
    torch.testing.assert_close(ids.cpu(), torch.full_like(expected_ids.cpu(), -1))
    torch.testing.assert_close(scores.cpu(), torch.full_like(expected_scores.cpu(), -torch.inf))
