# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Unit tests for the PCP owner long-KV stage cache."""

import torch

from vllm_ascend.attention.context_parallel import pcp_owner_stage as stage


def _table() -> torch.Tensor:
    return torch.tensor(
        [
            [3, 7, -1, -1],
            [0, 1, 2, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )


def test_build_stage_tables_dense_and_order():
    table = _table()
    staged, flat, counts = stage.build_stage_tables(table)
    assert counts.tolist() == [2, 3, 0]
    assert flat.tolist() == [3, 7, 0, 1, 2]
    assert staged.tolist() == [[0, 1, -1, -1], [2, 3, 4, -1], [-1, -1, -1, -1]]


def test_build_stage_tables_bounds_zero_filled_tail():
    # The scheduler zero-fills unused block-table slots; seq_lens must bound
    # the live prefix instead of trusting the padding value.
    table = torch.tensor([[5, 6, 7, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([130])
    staged, flat, counts = stage.build_stage_tables(table, seq_lens=seq_lens, block_size=64)
    assert counts.tolist() == [3]
    assert flat.tolist() == [5, 6, 7]
    assert staged.tolist() == [[0, 1, 2, -1, -1]]


def test_fixed_tables_match_mask_path():
    table = _table()
    counts = [2, 3, 0]
    staged, flat, total = stage.build_stage_tables_fixed(table, counts)
    ref_staged, ref_flat, _ = stage.build_stage_tables(table)
    assert staged.tolist() == ref_staged.tolist()
    assert flat.tolist() == ref_flat.tolist()
    assert total == 5


def test_fixed_tables_clamp_negative_padding():
    # An over-approximated count may cover a -1 padding slot; it must copy a
    # valid block instead of addressing garbage.
    table = torch.tensor([[4, -1, 0, 0]], dtype=torch.int32)
    staged, flat, total = stage.build_stage_tables_fixed(table, [2])
    assert flat.tolist() == [4, 0]
    assert total == 2
    assert staged.tolist() == [[0, 1, -1, -1]]


def test_fixed_tables_clamp_stale_high_ids():
    # A stale padding entry above the block range must not issue an
    # out-of-bounds gather.
    table = torch.tensor([[3, 99999, 0, 0]], dtype=torch.int32)
    _, flat, _ = stage.build_stage_tables_fixed(table, [2], num_blocks=8)
    assert flat.tolist() == [3, 7]


def test_stage_cache_matches_original_rows():
    table = _table()
    source = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    cache = stage.PcpOwnerStageCache(torch.device("cpu"))
    staged_source, staged_table = cache.stage(source, table)
    for req in range(table.shape[0]):
        for index in range(table.shape[1]):
            original = int(table[req, index])
            if original < 0:
                assert int(staged_table[req, index]) < 0
                continue
            slot = int(staged_table[req, index])
            assert torch.equal(staged_source[slot], source[original])


def test_stage_cache_overflow_falls_back(monkeypatch):
    monkeypatch.setattr(stage, "_stage_budget_bytes", lambda: 3 * 3 * 4)
    source = torch.zeros(8, 3)
    cache = stage.PcpOwnerStageCache(torch.device("cpu"))
    small = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
    assert cache.stage(source, small) is not None
    wide = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    assert cache.stage(source, wide) is None
    assert cache.overflow_fallbacks == 1


def test_stage_cache_empty_table():
    cache = stage.PcpOwnerStageCache(torch.device("cpu"))
    empty = torch.full((2, 4), -1, dtype=torch.int32)
    assert cache.stage(torch.zeros(8, 3), empty) is None


def test_get_stage_cache_keys_by_block_shape():
    stage._stage_caches.clear()
    wide = torch.zeros(8, 4)
    narrow = torch.zeros(8, 2)
    device = torch.device("cpu")
    first = stage.get_stage_cache(device, wide)
    assert stage.get_stage_cache(device, wide) is first
    second = stage.get_stage_cache(device, narrow)
    assert second is not first
    stage._stage_caches.clear()


def test_stage_enabled_env(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_PCP_OWNER_STAGE", raising=False)
    assert stage.owner_stage_enabled() is False
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_STAGE", "1")
    assert stage.owner_stage_enabled() is True
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_STAGE", "off")
    assert stage.owner_stage_enabled() is False
