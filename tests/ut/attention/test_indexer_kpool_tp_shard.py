# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP query-row sharding of the GLM KPool indexer (see AscendIndexerKPoolRowShard):
each rank scores a contiguous slice of the batch against the full visible key
history, and the group exchanges the packed int32 top-k rows in place."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.indexer_kpool import (
    MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK,
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolRowShard,
    _exchange_row_shard_topk,
    balanced_indexer_row_shard,
)

POOL_SIZE = 4
MODULE = "vllm_ascend.attention.indexer_kpool"


def _common_metadata(query_lens: list[int], seq_lens: list[int]) -> SimpleNamespace:
    """Minimal CommonAttentionMetadata for host-side shard planning."""
    boundaries = [0]
    for query_len in query_lens:
        boundaries.append(boundaries[-1] + query_len)
    num_reqs = len(query_lens)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=boundaries[-1],
        query_start_loc_cpu=torch.tensor(boundaries, dtype=torch.int64),
        # None mirrors async spec-decode mode: planning must fall back to
        # seq_lens_cpu (and, if that is None too, to a seq_lens host copy).
        _seq_lens_cpu=None,
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        positions=torch.arange(boundaries[-1], dtype=torch.int64),
        block_table_tensor=torch.arange(num_reqs * 8, dtype=torch.int64).reshape(num_reqs, 8),
    )


def _builder(tp_size: int = 4, **overrides) -> AscendIndexerKPoolMetadataBuilder:
    """Builder with only the fields shard planning touches."""
    builder = AscendIndexerKPoolMetadataBuilder.__new__(AscendIndexerKPoolMetadataBuilder)
    builder.tp_size = tp_size
    builder.pcp_size = overrides.pop("pcp_size", 1)
    builder.compress_ratio = POOL_SIZE
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=overrides.pop("decode_context_parallel_size", 1),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=overrides.pop("cudagraph_mode", CUDAGraphMode.FULL_DECODE_ONLY),
        ),
    )
    assert not overrides, f"unknown overrides: {overrides}"
    return builder


def test_balanced_row_shard_partitions_and_declines():
    """The split covers every row once with >= 1 row per rank and declines
    below the activation floor (or on a single rank)."""
    for tp_size in (2, 4, 8):
        rows = MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK * tp_size
        shapes = [
            ([4 * rows], [4 * rows]),  # fresh prompt: the full causal cost ramp
            # ragged requests at different context depths
            ([2 * rows, 7, 4 * rows, rows // 3 + 5], [2 * rows, 9007, 4 * rows + 500, rows // 3 + 60005]),
        ]
        for query_lens, seq_lens in shapes:
            sizes = balanced_indexer_row_shard(
                torch.tensor(seq_lens, dtype=torch.int32),
                torch.tensor(query_lens, dtype=torch.int32),
                POOL_SIZE,
                tp_size,
                MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK,
            )
            assert sizes is not None and len(sizes) == tp_size
            assert sum(sizes) == sum(query_lens)
            assert min(sizes) >= 1
            # Contiguous cost balance: each slice stays within one row's
            # scoring cost of its fair share (per-row reference cost).
            q = torch.tensor(query_lens, dtype=torch.int64)
            s = torch.tensor(seq_lens, dtype=torch.int64)
            first_key = torch.repeat_interleave(s - q + 1, q)
            row_idx = torch.arange(int(q.sum()), dtype=torch.int64)
            row_in_req = row_idx - torch.repeat_interleave(torch.cumsum(q, 0) - q, q)
            row_costs = (first_key + row_in_req) // POOL_SIZE
            costs = []
            offset = 0
            for size in sizes:
                costs.append(int(row_costs[offset : offset + size].sum()))
                offset += size
            fair = sum(costs) / tp_size
            worst_row_cost = int(row_costs.max())
            assert max(costs) <= fair + worst_row_cost
    lens = torch.tensor([MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK - 1], dtype=torch.int32)
    assert balanced_indexer_row_shard(lens, lens, POOL_SIZE, 4, MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK) is None
    assert balanced_indexer_row_shard(lens, lens, POOL_SIZE, 1, MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK) is None


def test_row_shard_gate_rejects_unsupported_configurations():
    """The gate is the whole safety envelope; nothing else guards the exchange."""
    rows = MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK * 4
    assert _builder()._indexer_row_sharding_supported(rows, graph_capture=False)

    assert not _builder(tp_size=1)._indexer_row_sharding_supported(rows, graph_capture=False)
    assert not _builder(pcp_size=2)._indexer_row_sharding_supported(rows, graph_capture=False)
    assert not _builder(decode_context_parallel_size=2)._indexer_row_sharding_supported(rows, graph_capture=False)
    assert not _builder()._indexer_row_sharding_supported(rows, graph_capture=True)
    assert not _builder(cudagraph_mode=CUDAGraphMode.FULL)._indexer_row_sharding_supported(rows, graph_capture=False)
    assert not _builder()._indexer_row_sharding_supported(rows - 1, graph_capture=False)


def test_row_shard_slices_tile_rows_and_rebase_requests():
    """Across ranks the slices tile every row exactly once, the rebased
    request metadata satisfies CANN's T1 constraint, and a request cut by a
    boundary keeps the visible length of its last covered row."""
    query_lens = [5000, 1, 6000, 13000]
    seq_lens = [5000, 4097, 41000, 13002]
    tp_size = 4
    common = _common_metadata(query_lens, seq_lens)
    builder = _builder(tp_size=tp_size)
    covered = []
    for rank in range(tp_size):
        with patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=rank):
            shard = builder._build_indexer_row_shard(common, graph_capture=False)
        assert shard is not None
        assert shard.sizes[rank] == shard.stop - shard.start
        covered.append((shard.start, shard.stop))
        # CANN's TND layout requires the query prefix sum's last element
        # to equal the local row count.
        assert int(shard.cum_query_lens[-1]) == shard.stop - shard.start
        assert shard.positions.shape[0] == shard.stop - shard.start
        # Positions keep global values for per-row index packing.
        assert torch.equal(shard.positions, common.positions[shard.start : shard.stop])
        # The block table keeps only the slice's active requests.
        local_rows = int(shard.cum_query_lens[-1])
        for position in (shard.positions[0], shard.positions[-1]):
            request = int(torch.searchsorted(common.query_start_loc_cpu[1:], position + 1))
            assert common.block_table_tensor[request] in shard.block_table
        assert local_rows == sum(int(b) - int(a) for a, b in zip([0, *shard.cum_query_lens[:-1]], shard.cum_query_lens))
    covered.sort()
    assert [start for start, _ in covered] == [0, *[stop for _, stop in covered][:-1]]
    assert covered[-1][1] == common.num_actual_tokens


def test_row_shard_rebases_a_request_cut_midway():
    """A single request split across two ranks reports, per rank, the visible
    length of the rank's last covered row -- never a truncated history."""
    rows = 2 * MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK
    common = _common_metadata([rows], [rows])
    builder = _builder(tp_size=2)
    with patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0):
        head = builder._build_indexer_row_shard(common, graph_capture=False)
    with patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=1):
        tail = builder._build_indexer_row_shard(common, graph_capture=False)
    assert head is not None and tail is not None
    # Contiguous tiling; the boundary follows the cost ramp, not row parity.
    assert head.start == 0 and tail.stop == rows and head.stop == tail.start
    head_rows = head.stop - head.start
    tail_rows = tail.stop - tail.start
    # Rank 0's single sliced request ends at its last covered row.
    assert torch.equal(head.cum_query_lens, torch.tensor([head_rows], dtype=torch.int64))
    assert torch.equal(head.seq_lens, torch.tensor([head_rows // POOL_SIZE], dtype=torch.int64))
    assert torch.equal(head.pool_tail, torch.tensor([head_rows % POOL_SIZE], dtype=torch.int64))
    # Rank 1's last row sees the full history.
    assert torch.equal(tail.cum_query_lens, torch.tensor([tail_rows], dtype=torch.int64))
    assert torch.equal(tail.seq_lens, torch.tensor([rows // POOL_SIZE], dtype=torch.int64))
    assert torch.equal(tail.pool_tail, torch.tensor([rows % POOL_SIZE], dtype=torch.int64))


def test_exchange_pads_slices_and_restores_global_row_order():
    """The padded all-gather reassembles per-rank slices in place; trailing
    padding rows of the output buffer are never touched."""
    sizes = (5, 2, 4)
    total = sum(sizes)
    width = 6
    expected = torch.arange(total * width, dtype=torch.int32).reshape(total, width)
    max_rows = max(sizes)
    # Simulate the collective: every rank contributes its padded slice.
    padded = torch.full((len(sizes), max_rows, width), -1, dtype=torch.int32)
    for rank, size in enumerate(sizes):
        start = sum(sizes[:rank])
        padded[rank, :size] = expected[start : start + size]
    gathered = padded.reshape(len(sizes) * max_rows, width)

    shards = []
    for rank in range(len(sizes)):
        start = sum(sizes[:rank])
        stop = start + sizes[rank]
        dummy = torch.empty(0)
        shards.append(
            AscendIndexerKPoolRowShard(
                start=start,
                stop=stop,
                sizes=sizes,
                positions=dummy,
                cum_query_lens=dummy,
                seq_lens=dummy,
                pool_tail=dummy,
                block_table=dummy,
            )
        )

    with patch(f"{MODULE}.get_tp_group") as group:
        group.return_value.all_gather = lambda local, dim: gathered
        for rank, shard in enumerate(shards):
            buffer = torch.full((total + 3, width), -7, dtype=torch.int32)
            # Packing has already written this rank's slice.
            start = sum(sizes[:rank])
            buffer[start : start + sizes[rank]] = expected[start : start + sizes[rank]]
            _exchange_row_shard_topk(buffer, shard, width)
            assert torch.equal(buffer[:total], expected)
            assert torch.all(buffer[total:] == -7), "padding rows must stay outside the exchange"


def test_row_shard_falls_back_when_cpu_mirrors_absent():
    """Async spec-decode batches carry neither CPU seq_lens mirror; planning
    must fall back to a host copy of the device seq_lens."""
    rows = MIN_GLM53_INDEXER_TP_SHARD_ROWS_PER_RANK * 4
    common = _common_metadata([rows], [rows])
    common.seq_lens_cpu = None
    common._seq_lens_cpu = None
    with patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0):
        shard = _builder()._build_indexer_row_shard(common, graph_capture=False)
    assert shard is not None
    local_rows = shard.stop - shard.start
    assert torch.equal(shard.seq_lens, torch.tensor([local_rows // POOL_SIZE], dtype=torch.int64))


def test_build_indexer_row_shard_declines_below_floor():
    """Below the row floor planning is a no-op and the replicated path stays
    untouched."""
    short = _common_metadata([16], [16])
    assert _builder()._build_indexer_row_shard(short, graph_capture=False) is None
