# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 CP causal-query boundaries and native CANN orchestration on CPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import vllm_ascend.attention.indexer_kpool as backend_module
from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolTailMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool_cann import SparseAttnIndexerKpool


@pytest.fixture(autouse=True)
def cpu_metadata_kernels(monkeypatch):
    from tests.ut.helpers.golden_keypool_metadata import build_keypool_metadata, build_tail_inputs
    from vllm_ascend.ops.triton import glm5_next_cann_keypool as kernels

    monkeypatch.setattr(kernels, "build_keypool_metadata", build_keypool_metadata)
    monkeypatch.setattr(kernels, "build_tail_inputs", build_tail_inputs)


class _DeviceOnlyCommon(SimpleNamespace):
    @property
    def _seq_lens_cpu(self):
        raise AssertionError("FP8 CP must not read host sequence lengths")


def _metadata(monkeypatch, rank, *, actual=12, graph=False, lookahead=5):
    monkeypatch.setattr(backend_module, "enable_dsa_cp", lambda: True)
    monkeypatch.setattr(backend_module, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=rank))
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        attention_config=SimpleNamespace(indexer_kv_dtype="fp8"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=8192),
    )
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        tokens_per_state=4,
        model_version="glm5_next",
        scale_dim=1,
        scale_dtype=torch.float32,
    )
    tail_spec = AscendIndexerKPoolTailSpec(
        block_size=4 + lookahead,
        sliding_window=4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
    )
    builder = AscendIndexerKPoolMetadataBuilder(spec, ["layer.indexer.k_cache"], config, torch.device("cpu"))
    tail_builder = AscendIndexerKPoolTailMetadataBuilder(
        tail_spec, ["layer.indexer.tail_cache"], config, torch.device("cpu")
    )
    common = _DeviceOnlyCommon(
        num_reqs=2,
        num_input_tokens=12,
        num_actual_tokens=actual,
        max_query_len=7,
        query_start_loc=torch.tensor([0, 7, actual], dtype=torch.int32),
        seq_lens=torch.tensor([21, 18 if actual > 7 else 0], dtype=torch.int32),
        positions=torch.tensor(list(range(14, 21)) + list(range(13, 18))),
        slot_mapping=torch.tensor(list(range(14, 21)) + list(range(141, 146))),
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    if actual < 12:
        common.slot_mapping[actual:] = -1
    build = builder.build_for_graph_capture if graph else lambda c: builder.build(0, c)
    metadata = build(common)
    tail = tail_builder.build(0, common)
    return metadata, tail, common, builder, tail_builder


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("lookahead", [0, 5])
def test_fp8_cp_causal_lengths_are_local_and_cache_writes_are_global(monkeypatch, rank, graph, lookahead):
    metadata, tail, common, builder, tail_builder = _metadata(monkeypatch, rank, graph=graph, lookahead=lookahead)
    query = metadata.query_metadata
    expected_raw = []
    for req, (start, end) in enumerate(zip(common.query_start_loc[:-1].tolist(), common.query_start_loc[1:].tolist())):
        rows = list(range(max(start, rank * 3), min(end, (rank + 1) * 3)))
        expected_raw.append(int(common.positions[rows[-1]]) + 1 if rows else 0)
    assert (query.seq_lens * 4 + query.pool_tail).tolist() == expected_raw
    assert query.positions.tolist() == common.positions[rank * 3 : (rank + 1) * 3].tolist()
    assert metadata.cum_query_lens.tolist() == [7, 12]
    assert metadata.seq_lens.tolist() == [5, 4]
    assert metadata.keypool_state_table is not None
    assert tail.cann_inputs is not None
    assert (tail.cann_inputs.retained_indices is not None) == (lookahead > 0)
    first = [query.cum_query_lens.data_ptr(), query.seq_lens.data_ptr(), query.pool_tail.data_ptr()]
    common.seq_lens += 4
    common.positions += 4
    refreshed = builder.build_for_graph_capture(common).query_metadata
    assert first == [refreshed.cum_query_lens.data_ptr(), refreshed.seq_lens.data_ptr(), refreshed.pool_tail.data_ptr()]
    expected = [length + 4 if length else 0 for length in expected_raw]
    assert (refreshed.seq_lens * 4 + refreshed.pool_tail).tolist() == expected
    assert tail_builder.build_for_drafting(common, 1).cann_inputs is not None


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("compute_topk", [False, True])
@pytest.mark.parametrize("actual", [7, 12])
@pytest.mark.parametrize("skip_empty", [False, True])
def test_cann_fp8_cp_scores_only_local_queries_without_future_pools(
    monkeypatch, rank, compute_topk, actual, skip_empty
):
    metadata, tail_metadata, common, _, _ = _metadata(monkeypatch, rank, actual=actual)
    query_metadata = metadata.query_metadata
    # Distinct values make an erroneously exposed future pool win top-k.
    cache = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32)
    cache = cache.view(2, 4, 1, 1).expand(2, 4, 1, 128).to(torch.float8_e4m3fn)
    scales = torch.ones(2, 4, 1, 1)
    op = SparseAttnIndexerKpool.__new__(SparseAttnIndexerKpool)
    nn.Module.__init__(op)
    op.topk_tokens, op.head_dim = 4, 128
    op._prepare_keypool_state = MagicMock(return_value=torch.zeros(5, 4, 256))
    op._key_pool = MagicMock(return_value=torch.zeros(3, 128))
    op._gather_retained_hidden = lambda hidden, indices: hidden.index_select(0, indices.flatten())
    op._write_keypool_caches = MagicMock()
    op._quantize_keypool_query = MagicMock(side_effect=lambda q: (q.to(torch.float8_e4m3fn), torch.ones(q.shape[:2])))
    calls = []

    def select(query, keys, weights, pool_tail, **kwargs):
        assert keys is cache and kwargs["k_descale"].data_ptr() == scales.data_ptr()
        assert kwargs["actual_seq_q"].dtype == kwargs["actual_seq_k"].dtype == torch.int64
        assert kwargs["quant_mode"] == 0 and kwargs["mask_mode"] == 3
        ends = kwargs["actual_seq_q"].tolist()
        lengths = kwargs["actual_seq_k"].tolist()
        result = torch.full((query.shape[0], 7), -1, dtype=torch.int32)
        start = 0
        for req, end in enumerate(ends):
            raw_end = lengths[req] * 4 + int(pool_tail[req])
            for row in range(start, end):
                position = raw_end - (end - start) + row - start
                assert position == int(query_metadata.positions[row])
                visible = (position + 1) // 4
                if visible:
                    # The test cache has four pools; use a larger logical cache
                    # for old-prefix rows without reading unallocated storage.
                    selected_pool = visible - 1
                    result[row, :4] = selected_pool * 4 + torch.arange(4)
            start = end
        calls.append(kwargs)
        return result, torch.empty(0)

    op._pool_key_indexer = select

    def pack(indices, score_metadata, topk, pool, output):
        assert score_metadata.positions is query_metadata.positions
        assert score_metadata.cum_query_lens.tolist() == query_metadata.cum_query_lens.tolist()
        output[: indices.shape[0], :7].copy_(indices)
        return output[: indices.shape[0], :7].unsqueeze(1)

    op._pack_keypool_indices = pack
    output = torch.empty(3, 7, dtype=torch.int32)
    result = op(
        torch.zeros(12, 256, dtype=torch.bfloat16),
        torch.ones(3, 1, 128, dtype=torch.bfloat16) if compute_topk else None,
        torch.ones(3, 1, dtype=torch.bfloat16) if compute_topk else None,
        metadata.positions,
        cache,
        torch.zeros(2, 9, 256),
        metadata,
        tail_metadata,
        key_weight=torch.zeros(128, 256, dtype=torch.bfloat16),
        gate_weight=torch.zeros(128, 256, dtype=torch.bfloat16),
        norm_weight=None,
        norm_bias=None,
        norm_eps=1e-6,
        compress_ape=torch.zeros(4, 128),
        index_kpool=4,
        compute_topk=compute_topk,
        output_buffer=output,
        indexer_scale_cache=scales,
        skip_empty_cp_scoring=skip_empty,
    )
    assert op._key_pool.call_args.args[0].shape[0] == 12
    assert op._write_keypool_caches.call_args.args[5] is metadata
    assert op._write_keypool_caches.call_args.kwargs["scale_cache"] is scales
    empty = query_metadata.num_actual_tokens == 0
    assert bool(calls) == (compute_topk and not (skip_empty and empty))
    if compute_topk and skip_empty and empty:
        assert torch.all(result == -1)
        op._quantize_keypool_query.assert_not_called()
    assert op._write_keypool_caches.call_count == 1
    if compute_topk:
        assert result.shape == (3, 1, 7)
        assert result.data_ptr() == output.data_ptr()
        assert torch.all(result[:, 0, :4] <= query_metadata.positions[:, None])
    else:
        assert result is None


@pytest.mark.parametrize("rank", [2, 3])
def test_fp8_cp_empty_local_requests_and_padding_expose_no_keys(monkeypatch, rank):
    metadata, _, _, _, _ = _metadata(monkeypatch, rank, actual=7, graph=True)
    query = metadata.query_metadata
    assert query.seq_lens[1] == query.pool_tail[1] == 0
    if rank == 3:
        assert query.num_actual_tokens == 0
        assert torch.count_nonzero(query.seq_lens) == 0
        assert torch.count_nonzero(query.pool_tail) == 0
    assert metadata.keypool_state_table is not None
