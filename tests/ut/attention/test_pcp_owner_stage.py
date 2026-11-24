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


def test_stage_cache_overflow_falls_back():
    source = torch.zeros(8, 3)
    cache = stage.PcpOwnerStageCache(torch.device("cpu"), budget_bytes=3 * 3 * 4)
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


def test_compressed_stage_counts_cover_cached_prefix():
    # A prefix-cache hit arrives as a large already-computed prefix: the stage
    # plan must select blocks for the whole live prefix, not just new tokens.
    seq_lens_cpu = torch.tensor([40000, 20001])
    counts = stage.compressed_stage_counts(seq_lens_cpu, ratio=2, block_size=128)
    assert counts == [157, 79]


def test_compressed_stage_counts_ratio_one_is_token_blocks():
    counts = stage.compressed_stage_counts([0, 1, 128, 129], ratio=1, block_size=128)
    assert counts == [0, 1, 1, 2]


def test_compressed_stage_counts_partial_group_rounds_up():
    # An unfinished compression group is still one live storage row, so the
    # count rounds up; padding rows are never referenced by the kernel.
    counts = stage.compressed_stage_counts([255, 256, 258], ratio=2, block_size=128)
    assert counts == [1, 1, 2]


def test_compressed_stage_counts_accepts_numpy_like_sequences():
    assert stage.compressed_stage_counts((100, 200), ratio=2, block_size=64) == [1, 2]
    assert stage.compressed_stage_counts([0], ratio=1, block_size=0) is None
    assert stage.compressed_stage_counts(None, ratio=2, block_size=128) is None


def test_mixed_prefill_decode_rows_select_whole_live_prefix():
    # Row 0 is a cached-prefix prefill (seq 40000, ratio 2 => 157 blocks); row 1
    # is a one-token decode whose history must still be staged in full.
    counts = stage.compressed_stage_counts(torch.tensor([40000, 30000]), ratio=2, block_size=128)
    table = torch.zeros((2, 157), dtype=torch.int32)
    table[0] = torch.arange(157, dtype=torch.int32)
    table[1, :118] = torch.arange(100, 100 + 118, dtype=torch.int32)
    staged, flat, total = stage.build_stage_tables_fixed(table, counts, num_blocks=512)
    assert total == 157 + 118
    assert staged[0, 156].item() == 156
    assert staged[1, 117].item() == 157 + 117
    # A decode row's staging table is dense over its whole history.
    assert torch.equal(staged[1, :118], torch.arange(157, 157 + 118, dtype=torch.int32))
    assert (staged[1, 118:] < 0).all()


def test_stage_cache_serves_mixed_rows_from_local_scratch():
    # End-to-end CPU check: a long cached-prefix row and a decode row are both
    # served from the scratch copy, at the slots their staged table names.
    table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, -1, -1],
        ],
        dtype=torch.int32,
    )
    source = torch.arange(32 * 3, dtype=torch.float32).reshape(32, 3)
    cache = stage.PcpOwnerStageCache(torch.device("cpu"))
    staged_source, staged_table = cache.stage(source, table)
    assert staged_table.tolist() == [[0, 1, 2, 3], [4, 5, -1, -1]]
    for row, column, original in ((0, 3, 13), (1, 1, 21)):
        slot = int(staged_table[row, column])
        assert torch.equal(staged_source[slot], source[original])


def test_swa_chunk_staging_supports_replay_visible_lengths():
    # Bounded replay stamps ori_topk_length, a right-edge visible count.  It is
    # shift invariant, so replay rows are stageable; explicit index lists are
    # not, because staging rebases their coordinates.
    assert stage.swa_chunk_staging_supported(None, None, 128, 127) is True
    assert stage.swa_chunk_staging_supported(None, torch.ones(4, 1, dtype=torch.int32), 128, 127) is True
    assert stage.swa_chunk_staging_supported(torch.zeros(4, 8, dtype=torch.int32), None, 128, 127) is False
    assert stage.swa_chunk_staging_supported(None, None, 0, 127) is False
    assert stage.swa_chunk_staging_supported(None, None, 128, -1) is False


def test_replay_visible_count_is_shift_invariant():
    # A replayed prefill of 5000 tokens keeps its last 128; the staged plan
    # starts at block-aligned 4736, so positions and the replay floor shift by
    # that offset.  The visible count (distance from the token's right edge)
    # must not change, otherwise the staged view would mask different KV.
    window, seq_len, staged_start = 128, 5000, 4736
    floor = seq_len - window
    assert staged_start <= floor

    def visible(pos, low):
        return pos - max(pos - window + 1, low) + 1

    assert visible(seq_len, floor) == window
    assert visible(seq_len, floor) == visible(seq_len - staged_start, floor - staged_start)
    # A token in the middle of the kept chunk must also keep its count.
    assert visible(4950, floor) == visible(4950 - staged_start, floor - staged_start)


def test_swa_chunk_plan_covers_replay_floor():
    # A replayed prefill keeps its last window: query_start_loc says 128 kept
    # rows, seq_lens says the chunk ends at 5000, so the logical floor is 4872.
    segments = stage.plan_swa_chunk_segments([0, 128], [5000], 1, 128, 128)
    assert segments is not None
    segment = segments[0]
    assert segment.replay_start <= 5000 - 128
    assert segment.replay_start + segment.replay_seq_len == 5000


def test_stage_enabled_uses_the_typed_config():
    from vllm_ascend.ascend_config import PcpConfig

    assert stage.owner_stage_enabled() is False
    assert stage.owner_stage_enabled(PcpConfig()) is False
    assert stage.owner_stage_enabled(PcpConfig(owner_stage=True)) is True
    assert stage.owner_stage_swa_chunk_enabled() is False
    assert stage.owner_stage_swa_chunk_enabled(PcpConfig(owner_stage_swa_chunk=True)) is True
