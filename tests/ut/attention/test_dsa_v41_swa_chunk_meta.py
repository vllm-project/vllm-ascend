# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Bounded SWA staging: per-batch plan and per-signature operator metadata.

The NPU operator and the planning helpers are replaced with CPU stand-ins; the
tests assert how often the real work would run, not its numeric result.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import pcp_owner_stage

NUM_REQS = 2
BLOCK_SIZE = 4
WINDOW = 3
NUM_BLOCKS = 16
INDEX_TABLE = [[1, 2], [2, 3]]


def _swa_metadata(*, query_lens=(3, 3), seqs=(10, 14), win_left=WINDOW - 1):
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    seq_lens = torch.tensor(seqs, dtype=torch.int32)
    return SimpleNamespace(
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seqs, dtype=torch.int32),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=seq_lens,
        max_query_len=max(query_lens),
        max_cache_seq_len=int(seq_lens.max()),
        ori_topk_length=None,
        ori_mask_mode=4,
        ori_win_left=win_left,
        ori_win_right=0,
    )


def _layer_table(offset=0):
    values = [offset + index for index in range(1, 9)]
    return torch.tensor([values[:4], values[4:]], dtype=torch.int32)


def _operator_metadata(max_cache_seq_len=14):
    return SimpleNamespace(max_cache_seq_len=max_cache_seq_len)


def _attn(n_local_heads=4, head_dim=8):
    return SimpleNamespace(n_local_heads=n_local_heads, head_dim=head_dim)


def _plan(swa, table, stage_cache, window=WINDOW):
    return dsa_v41._build_swa_chunk_plan(
        swa,
        table,
        NUM_REQS,
        BLOCK_SIZE,
        window,
        NUM_BLOCKS,
        stage_cache=stage_cache,
    )


def _smla_metadata(stage_cache, plan, swa=None, attn=None, operator_metadata=None, cmp_topk=8):
    return dsa_v41._build_swa_chunk_smla_metadata(
        attn if attn is not None else _attn(),
        swa if swa is not None else _swa_metadata(),
        operator_metadata if operator_metadata is not None else _operator_metadata(),
        plan,
        stage_cache=stage_cache,
        has_compressed=True,
        ratio=1,
        cmp_seq_lens=torch.zeros(NUM_REQS, dtype=torch.int32),
        cmp_residual=None,
        cmp_topk=cmp_topk,
        num_reqs=NUM_REQS,
    )


def _install_op(monkeypatch, side_effect):
    op = Mock(side_effect=side_effect)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", op, raising=False)
    return op


def _raise_operator_error(*_args, **_kwargs):
    raise RuntimeError("no operator")


def test_segment_plan_runs_once_per_batch_but_each_layer_gathers(monkeypatch):
    plan_calls = []
    build_calls = []
    original_plan = pcp_owner_stage.plan_swa_chunk_segments
    original_build = pcp_owner_stage.build_stage_tables_fixed

    def counted_plan(*args, **kwargs):
        plan_calls.append(args)
        return original_plan(*args, **kwargs)

    def counted_build(*args, **kwargs):
        build_calls.append(args)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(pcp_owner_stage, "plan_swa_chunk_segments", counted_plan)
    monkeypatch.setattr(pcp_owner_stage, "build_stage_tables_fixed", counted_build)

    stage_cache = dsa_v41._SwaChunkStageCache()
    swa = _swa_metadata()
    tables = [_layer_table(0), _layer_table(10), _layer_table(20), _layer_table(30)]
    plans = [_plan(swa, table, stage_cache) for table in tables]

    # One CPU segment plan for the whole batch, one gather per cache group.
    assert len(plan_calls) == 1
    assert len(build_calls) == 4
    assert all(plan is not None for plan in plans)
    for plan, table in zip(plans, tables):
        expected = torch.tensor(
            [[int(table[row, column]) for column in row_index] for row, row_index in enumerate(INDEX_TABLE)],
            dtype=table.dtype,
        )
        assert torch.equal(plan.window_block_table, expected)
        assert plan.counts == [2, 2]
        assert plan.replay_seq_lens.tolist() == [6, 6]
        assert plan.max_replay_seq_len == 6
    # Only the block-table contents differ per layer, not the plan geometry.
    assert plans[0].segments is plans[3].segments


def test_new_cpu_geometry_replans(monkeypatch):
    plan_calls = []
    original_plan = pcp_owner_stage.plan_swa_chunk_segments

    def counted_plan(*args, **kwargs):
        plan_calls.append(args)
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(pcp_owner_stage, "plan_swa_chunk_segments", counted_plan)
    stage_cache = dsa_v41._SwaChunkStageCache()

    _plan(_swa_metadata(), _layer_table(0), stage_cache)
    _plan(_swa_metadata(seqs=(10, 18)), _layer_table(0), stage_cache)

    assert len(plan_calls) == 2


def test_identical_signatures_share_one_metadata_op(monkeypatch):
    op = _install_op(monkeypatch, lambda *_args, **_kwargs: torch.arange(4, dtype=torch.int32))
    stage_cache = dsa_v41._SwaChunkStageCache()
    plan = _plan(_swa_metadata(), _layer_table(0), stage_cache)

    results = [_smla_metadata(stage_cache, plan) for _ in range(4)]

    assert op.call_count == 1
    assert all(result is not None for result in results)
    assert len({result.data_ptr() for result in results}) == 4
    for result in results[1:]:
        assert torch.equal(result, results[0])


def test_cached_metadata_is_cloned_from_a_reused_buffer(monkeypatch):
    shared_buffer = torch.arange(4, dtype=torch.int32)
    op = _install_op(monkeypatch, lambda *_args, **_kwargs: shared_buffer)
    stage_cache = dsa_v41._SwaChunkStageCache()
    plan = _plan(_swa_metadata(), _layer_table(0), stage_cache)

    results = [_smla_metadata(stage_cache, plan) for _ in range(2)]

    assert op.call_count == 1
    assert all(result.data_ptr() != shared_buffer.data_ptr() for result in results)
    assert results[0].data_ptr() != results[1].data_ptr()
    results[0].fill_(-1)
    assert shared_buffer.tolist() == [0, 1, 2, 3]
    assert results[1].tolist() == [0, 1, 2, 3]


def test_differing_signatures_rebuild_and_do_not_share(monkeypatch):
    counter = {"calls": 0}

    def output(*_args, **_kwargs):
        counter["calls"] += 1
        return torch.full((4,), counter["calls"], dtype=torch.int32)

    op = _install_op(monkeypatch, output)
    stage_cache = dsa_v41._SwaChunkStageCache()
    plan = _plan(_swa_metadata(), _layer_table(0), stage_cache)

    variants = [
        _smla_metadata(stage_cache, plan),
        _smla_metadata(stage_cache, plan, swa=_swa_metadata(win_left=WINDOW)),
        _smla_metadata(stage_cache, plan, operator_metadata=_operator_metadata(max_cache_seq_len=15)),
        _smla_metadata(stage_cache, plan, attn=_attn(n_local_heads=6)),
    ]

    assert op.call_count == 4
    assert [int(result[0]) for result in variants] == [1, 2, 3, 4]
    # Repeating the first signature still hits its own cached entry.
    assert int(_smla_metadata(stage_cache, plan)[0]) == 1
    assert op.call_count == 4


def test_segment_geometry_is_part_of_the_signature(monkeypatch):
    op = _install_op(monkeypatch, lambda *_args, **_kwargs: torch.zeros(4, dtype=torch.int32))
    stage_cache = dsa_v41._SwaChunkStageCache()

    # A wider table keeps both absolute block ranges in bounds; the point of
    # this test is the signature, not the table guard.
    wide_table = torch.arange(1, 17, dtype=torch.int32).reshape(2, 8)
    first = _plan(_swa_metadata(seqs=(10, 14)), wide_table, stage_cache)
    second = _plan(_swa_metadata(seqs=(10, 18)), wide_table, stage_cache)
    _smla_metadata(stage_cache, first)
    _smla_metadata(stage_cache, second)

    assert op.call_count == 2


def test_absolute_block_range_outside_the_table_fails_open():
    stage_cache = dsa_v41._SwaChunkStageCache()

    # The second row needs absolute blocks [3, 5) in a four-column table.
    assert _plan(_swa_metadata(seqs=(10, 18)), _layer_table(0), stage_cache) is None


def test_metadata_failure_fails_open_and_is_remembered(monkeypatch):
    op = _install_op(monkeypatch, _raise_operator_error)
    stage_cache = dsa_v41._SwaChunkStageCache()
    plan = _plan(_swa_metadata(), _layer_table(0), stage_cache)

    assert _smla_metadata(stage_cache, plan) is None
    assert _smla_metadata(stage_cache, plan) is None
    assert op.call_count == 1


def _layer_impl(swa_prefix="layer.0.swa_cache"):
    return SimpleNamespace(
        swa_prefix=swa_prefix,
        long_kv_source_prefix=None,
        index_k_source_prefix=None,
        compressor_state_prefix=None,
        role=SimpleNamespace(is_kv_source=False),
    )


def test_layer_metadata_reuses_the_builder_published_stage_cache():
    stage_cache = dsa_v41._SwaChunkStageCache()
    swa = _swa_metadata()
    swa.swa_stage_cache = stage_cache
    metadata = {"layer.0.swa_cache": swa}
    impl = _layer_impl()

    first = dsa_v41.AscendDSAV41Impl._get_layer_metadata(impl, metadata)
    second = dsa_v41.AscendDSAV41Impl._get_layer_metadata(impl, metadata)

    assert first.swa_stage_cache is stage_cache
    assert second.swa_stage_cache is stage_cache


def test_layer_metadata_falls_back_to_one_shared_cache():
    swa = _swa_metadata()
    metadata = {"layer.0.swa_cache": swa}
    impl = _layer_impl()

    first = dsa_v41.AscendDSAV41Impl._get_layer_metadata(impl, metadata)
    second = dsa_v41.AscendDSAV41Impl._get_layer_metadata(impl, metadata)

    assert first.swa_stage_cache is second.swa_stage_cache
    assert getattr(swa, dsa_v41._SWA_CHUNK_STAGE_CACHE_KEY) is first.swa_stage_cache
