# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer_kpool import KPoolIndexerGraphParams
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import ContextSource, UpdatableGraph
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton as select


@pytest.mark.parametrize("pool_capacity", [2048, 262144])
@torch.inference_mode()
def test_graph_pool_length_growth_shrink_and_padding(pool_capacity):
    """One captured graph must follow live lengths, including past 8K."""
    torch.manual_seed(17)
    rows, head_dim, block_size, topk = 18, 128, 128, 2048
    blocks = pool_capacity // block_size
    cache = torch.randn(blocks, block_size, 1, head_dim, dtype=torch.bfloat16, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").repeat(3, 1)
    query = torch.randn(rows, 32, head_dim, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(rows, 32, dtype=torch.bfloat16, device="npu")
    ends = torch.tensor([6, 12, 15], dtype=torch.int32, device="npu")
    lengths = torch.full((3,), 1024, dtype=torch.int32, device="npu")
    positions = torch.full((rows,), 4095, dtype=torch.int64, device="npu")
    kwargs = dict(
        index_topk=topk,
        index_kpool=4,
        max_pool_seq_len=pool_capacity,
        pack_tail=True,
        allow_cache_packing=False,
    )
    args = query, cache, weights, ends, lengths, table, positions
    select(*args, **kwargs)
    torch.npu.synchronize()
    graph = UpdatableGraph()
    provider = KPoolIndexerGraphParams("cache")
    with torch.npu.graph(graph):
        result = select(*args, **kwargs, graph_params=provider)
    assert len(graph.tasks) == 1
    address = result.data_ptr()
    update_stream = torch.npu.Stream()
    pool_lengths = [1024, 1025, 1536, 2048, 511, 0]
    if pool_capacity > 2048:
        pool_lengths += [2049, 8192, 262144, 1024]
    for count in pool_lengths:
        lengths.fill_(count)
        positions.fill_(max(0, count * 4 - 1))
        context = {"cache": SimpleNamespace(seq_lens_cpu=torch.tensor([count] * 3))}
        graph.update(update_stream, graph.resolve_tasks(ContextSource(context)))
        update_stream.synchronize()
        graph.replay()
        torch.npu.synchronize()
        # The unmodified eager scoring/TopK path at the same bounded width is
        # the oracle. Query weighting, scores and index expansion are shared.
        width = min(pool_capacity, 1 << (max(512, count) - 1).bit_length())
        expected = select(*args, **{**kwargs, "max_pool_seq_len": width})
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert result.data_ptr() == address
        assert (result[15:] == -1).all()
        if count:
            assert ((result[:15] < count * 4) | (result[:15] == -1)).all()


@torch.inference_mode()
def test_equal_live_lengths_are_context_limit_independent():
    torch.manual_seed(91)
    cache = torch.randn(2048, 128, 1, 128, dtype=torch.bfloat16, device="npu")
    table = torch.arange(2048, dtype=torch.int32, device="npu").view(1, -1)
    query = torch.randn(6, 32, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(6, 32, dtype=torch.bfloat16, device="npu")
    ends = torch.tensor([6], dtype=torch.int32, device="npu")
    lengths = torch.tensor([1536], dtype=torch.int32, device="npu")
    positions = torch.arange(6138, 6144, dtype=torch.int64, device="npu")
    args = query, cache, weights, ends, lengths, table, positions
    graphs, outputs = [], []
    provider = KPoolIndexerGraphParams("cache")
    for capacity in [2048, 262144]:
        kwargs = dict(index_topk=2048, index_kpool=4, max_pool_seq_len=capacity, allow_cache_packing=False)
        select(*args, **kwargs)
        torch.npu.synchronize()
        graph = UpdatableGraph()
        with torch.npu.graph(graph):
            outputs.append(select(*args, **kwargs, graph_params=provider))
        graphs.append(graph)
    update_stream = torch.npu.Stream()
    context = {"cache": SimpleNamespace(seq_lens_cpu=torch.tensor([1536]))}
    for graph in graphs:
        graph.update(update_stream, graph.resolve_tasks(ContextSource(context)))
        update_stream.synchronize()
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@torch.inference_mode()
def test_merged_mtp_graph_updates_all_five_steps():
    torch.manual_seed(99)
    cache = torch.randn(64, 128, 1, 128, dtype=torch.bfloat16, device="npu")
    table = torch.arange(64, dtype=torch.int32, device="npu").view(1, -1)
    query = torch.randn(3, 32, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(3, 32, dtype=torch.bfloat16, device="npu")
    ends = torch.tensor([3], dtype=torch.int32, device="npu")
    steps = [
        (
            query,
            cache,
            weights,
            ends,
            torch.tensor([1024 + i], dtype=torch.int32, device="npu"),
            table,
            torch.full((3,), (1024 + i) * 4 - 1, dtype=torch.int64, device="npu"),
        )
        for i in range(5)
    ]
    kwargs = dict(index_topk=2048, index_kpool=4, max_pool_seq_len=8192, allow_cache_packing=False)
    for args in steps:
        select(*args, **kwargs)
    torch.npu.synchronize()
    graph = UpdatableGraph()
    providers = [KPoolIndexerGraphParams("cache", context_index=i) for i in range(5)]
    with torch.npu.graph(graph):
        outputs = [select(*args, **kwargs, graph_params=provider) for args, provider in zip(steps, providers)]
    assert all(graph.provider_sizes[provider] == 1 for provider in providers)
    context = SimpleNamespace(
        draft_attn_metadatas=[
            {
                "cache": SimpleNamespace(
                    seq_lens_cpu=torch.tensor([1024]) if i == 0 else None,
                    pool_seq_len_upper_bound=1024 + i,
                )
            }
            for i in range(5)
        ]
    )
    wrapper = SimpleNamespace(update_stream=torch.npu.Stream(), attn_backend=None, enable_enpu=False)
    with patch("vllm_ascend.compilation.acl_graph._EXTRA_CTX", SimpleNamespace(is_draft_model=True)):
        # Exercise the serving wrapper's replay-then-update event ordering.
        ACLGraphWrapper._updatable_graph_replay(wrapper, context, graph)
    torch.npu.synchronize()
    for i, (args, output) in enumerate(zip(steps, outputs)):
        width = 1024 if i == 0 else 2048
        expected = select(*args, **{**kwargs, "max_pool_seq_len": width})
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
