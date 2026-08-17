from vllm_ascend.attention.context_parallel.compressor_sp import (
    all_ranks_have_compressor_sp_rows,
    build_compressor_sp_plan,
    collect_state_row_indices,
)


def _plan(
    ratio,
    positions,
    query_start_loc,
    seq_lens,
    local_start,
    local_end,
    allow_c4_non_aligned=False,
    allow_c128_non_aligned=False,
    is_chunked_prefill=False,
    tp_size=4,
    tp_rank=0,
):
    return build_compressor_sp_plan(
        enabled=True,
        has_prefill=True,
        need_gather_q_kv=True,
        tp_size=tp_size,
        compress_ratio=ratio,
        allow_c4_non_aligned=allow_c4_non_aligned,
        allow_c128_non_aligned=allow_c128_non_aligned,
        is_chunked_prefill=is_chunked_prefill,
        input_positions=positions,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        local_start=local_start,
        local_end=local_end,
        tp_rank=tp_rank,
    )


def test_c4_local_rank_uses_left_overlap_and_crops_to_owned_row():
    plan = _plan(
        ratio=4,
        positions=list(range(16)),
        query_start_loc=[0, 16],
        seq_lens=[16],
        local_start=4,
        local_end=8,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(0, 8))
    assert plan.token_slice == (0, 8)
    assert plan.req_indices == (0,)
    assert plan.req_slice == (0, 1)
    assert plan.cu_seqlens == (0, 8)
    assert plan.start_pos == (0,)
    assert plan.history_start_positions == (0,)
    assert plan.start_pos_zero
    assert not plan.requires_history_state
    assert plan.compressed_row_indices == (0, 1)
    assert plan.compressed_row_slice == (0, 2)
    assert plan.rope_row_indices == (0, 1, 0)
    assert plan.rope_row_slice is None
    assert plan.valid_row_indices == (1,)
    assert plan.valid_row_slice == (1, 1)
    assert plan.output_keep_indices == (1,)
    assert plan.output_keep_slice == (1, 1)
    assert plan.slot_mapping_indices == (1,)
    assert plan.slot_mapping_slice == (1, 1)
    assert plan.local_keep_to_full_row_indices == (1,)


def test_c128_local_rank_expands_to_full_compression_group():
    plan = _plan(
        ratio=128,
        positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=64,
        local_end=128,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(0, 128))
    assert plan.token_slice == (0, 128)
    assert plan.req_indices == (0,)
    assert plan.req_slice == (0, 1)
    assert plan.cu_seqlens == (0, 128)
    assert plan.start_pos == (0,)
    assert plan.compressed_row_indices == (0,)
    assert plan.valid_row_indices == (0,)
    assert plan.compressed_row_slice == (0, 1)
    assert plan.output_keep_indices == (0,)
    assert plan.output_keep_slice == (0, 1)
    assert plan.slot_mapping_indices == (0,)
    assert plan.slot_mapping_slice == (0, 1)


def test_c128_late_rank_expands_from_group_start():
    plan = _plan(
        ratio=128,
        positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=192,
        local_end=256,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(128, 256))
    assert plan.token_slice == (128, 128)
    assert plan.start_pos == (128,)
    assert plan.compressed_row_indices == (1,)
    assert plan.compressed_row_slice == (1, 1)
    assert plan.output_keep_indices == (0,)
    assert plan.output_keep_slice == (0, 1)
    assert plan.slot_mapping_indices == (1,)
    assert plan.slot_mapping_slice == (1, 1)
    assert plan.valid_row_indices == (0,)
    assert plan.padding_row_indices == (1,)


def test_multi_request_overlap_does_not_cross_request_boundary():
    plan = _plan(
        ratio=4,
        positions=list(range(8)) + list(range(8)),
        query_start_loc=[0, 8, 16],
        seq_lens=[8, 8],
        local_start=7,
        local_end=12,
    )

    assert plan.enabled
    assert plan.req_indices == (0, 1)
    assert plan.req_slice == (0, 2)
    assert plan.token_indices == tuple(range(16))
    assert plan.token_slice == (0, 16)
    assert plan.cu_seqlens == (0, 8, 16)
    assert plan.start_pos == (0, 0)
    assert plan.compressed_row_indices == (0, 1, 2, 3)
    assert plan.compressed_row_slice == (0, 4)
    assert plan.output_keep_indices == (1, 2)
    assert plan.output_keep_slice == (1, 2)
    assert plan.slot_mapping_indices == (1, 2)
    assert plan.slot_mapping_slice == (1, 2)
    assert plan.valid_row_indices == (1, 2)


def test_unaligned_final_seq_len_falls_back():
    plan = _plan(
        ratio=4,
        positions=list(range(10)),
        query_start_loc=[0, 10],
        seq_lens=[10],
        local_start=0,
        local_end=4,
    )

    assert not plan.enabled
    assert plan.reason == "seq_len_not_aligned_c4"


def test_c4_non_aligned_allowed_replicates_state_only_tail():
    plan = _plan(
        ratio=4,
        positions=list(range(42)),
        query_start_loc=[0, 42],
        seq_lens=[42],
        local_start=0,
        local_end=4,
        allow_c4_non_aligned=True,
    )

    assert plan.enabled
    assert not plan.seq_len_aligned
    assert plan.requires_tail_state_update
    assert plan.token_indices == tuple(range(8)) + tuple(range(32, 42))
    assert plan.token_slice is None
    assert plan.req_indices == (0, 0)
    assert plan.cu_seqlens == (0, 8, 18)
    assert plan.start_pos == (0, 32)
    assert plan.compressed_row_indices == (0, 1, 8, 9)
    assert plan.output_keep_indices == (0,)
    assert plan.slot_mapping_indices == (0,)
    assert plan.tail_token_ranges == ((40, 42),)
    assert plan.padding_row_indices == (4, 5)


def test_c128_non_aligned_falls_back_without_allow_flag():
    plan = _plan(
        ratio=128,
        positions=list(range(130)),
        query_start_loc=[0, 130],
        seq_lens=[130],
        local_start=0,
        local_end=130,
    )

    assert not plan.enabled
    assert plan.reason == "seq_len_not_aligned_c128"


def test_c128_non_aligned_allowed_keeps_valid_rows_and_marks_tail():
    plan = _plan(
        ratio=128,
        positions=list(range(130)),
        query_start_loc=[0, 130],
        seq_lens=[130],
        local_start=0,
        local_end=130,
        allow_c128_non_aligned=True,
    )

    assert plan.enabled
    assert plan.seq_len_aligned is False
    assert plan.requires_tail_state_update
    assert plan.compressed_row_indices == (0,)
    assert plan.valid_row_indices == (0,)
    assert plan.output_keep_indices == (0,)
    assert plan.slot_mapping_indices == (0,)
    assert plan.padding_row_indices == (1,)
    assert plan.tail_token_ranges == ((128, 130),)


def test_c128_non_aligned_keeps_tail_guard_when_tail_is_other_rank_owned():
    plan = _plan(
        ratio=128,
        positions=list(range(130)),
        query_start_loc=[0, 130],
        seq_lens=[130],
        local_start=0,
        local_end=128,
        allow_c128_non_aligned=True,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(130))
    assert plan.req_indices == (0, 0)
    assert plan.cu_seqlens == (0, 128, 130)
    assert plan.start_pos == (0, 128)
    assert plan.tail_token_ranges == ((128, 130),)
    assert plan.requires_tail_state_update


def test_c128_non_aligned_middle_rank_expands_right_to_complete_group():
    plan = _plan(
        ratio=128,
        positions=list(range(260)),
        query_start_loc=[0, 260],
        seq_lens=[260],
        local_start=64,
        local_end=194,
        allow_c128_non_aligned=True,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(260))
    assert plan.token_slice == (0, 260)
    assert plan.req_indices == (0, 0)
    assert plan.cu_seqlens == (0, 256, 260)
    assert plan.start_pos == (0, 256)
    assert plan.compressed_row_indices == (0, 1)
    assert plan.output_keep_indices == (0,)
    assert plan.slot_mapping_indices == (0,)
    assert plan.tail_token_ranges == ((256, 260),)
    assert plan.requires_tail_state_update


def test_c128_non_aligned_history_start_is_allowed_for_validation():
    plan = _plan(
        ratio=128,
        positions=list(range(130, 256)),
        query_start_loc=[0, 126],
        seq_lens=[256],
        local_start=0,
        local_end=126,
        allow_c128_non_aligned=True,
    )

    assert plan.enabled
    assert plan.start_pos == (130,)
    assert plan.request_start_positions == (130,)
    assert plan.requires_history_state
    assert plan.output_keep_indices == (0,)
    assert plan.slot_mapping_indices == (0,)


def test_c4_nonzero_start_pos_uses_absolute_local_start_pos():
    plan = _plan(
        ratio=4,
        positions=list(range(8, 16)),
        query_start_loc=[0, 8],
        seq_lens=[16],
        local_start=0,
        local_end=4,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(0, 8))
    assert plan.token_slice == (0, 8)
    assert plan.req_indices == (0,)
    assert plan.req_slice == (0, 1)
    assert plan.cu_seqlens == (0, 8)
    assert plan.start_pos == (8,)
    assert plan.history_start_positions == (8,)
    assert not plan.start_pos_zero
    assert plan.requires_history_state
    assert plan.compressed_row_indices == (0, 1)
    assert plan.compressed_row_slice == (0, 2)
    assert plan.output_keep_indices == (0,)
    assert plan.output_keep_slice == (0, 1)
    assert plan.slot_mapping_indices == (0,)
    assert plan.slot_mapping_slice == (0, 1)
    assert plan.local_keep_to_full_row_indices == (0,)


def test_c4_nonzero_start_pos_state_rows_use_absolute_positions():
    plan = _plan(
        ratio=4,
        positions=list(range(8, 16)),
        query_start_loc=[0, 8],
        seq_lens=[16],
        local_start=0,
        local_end=4,
    )

    rows = collect_state_row_indices(
        token_positions=[8, 9, 10, 11],
        req_block_table=[[1, 2, 3]],
        cu_seqlens=plan.cu_seqlens,
        state_block_size=8,
    )

    assert rows == (2,)


def test_state_rows_include_zero_based_physical_block_id():
    plan = _plan(
        ratio=4,
        positions=list(range(8)),
        query_start_loc=[0, 8],
        seq_lens=[8],
        local_start=0,
        local_end=4,
    )

    rows = collect_state_row_indices(
        token_positions=[0, 1, 2, 3],
        req_block_table=[[0, 1]],
        cu_seqlens=plan.cu_seqlens,
        state_block_size=4,
    )

    assert rows == (0,)


def test_invalid_query_start_loc_falls_back():
    plan = _plan(
        ratio=4,
        positions=list(range(8)),
        query_start_loc=[0, 9],
        seq_lens=[9],
        local_start=0,
        local_end=4,
    )

    assert not plan.enabled
    assert plan.reason == "query_start_loc_out_of_bounds"


def test_c4_nonzero_start_pos_expands_within_current_query():
    plan = _plan(
        ratio=4,
        positions=list(range(8, 24)),
        query_start_loc=[0, 16],
        seq_lens=[24],
        local_start=8,
        local_end=12,
    )

    assert plan.enabled
    assert plan.token_indices == tuple(range(0, 16))
    assert plan.token_slice == (0, 16)
    assert plan.req_indices == (0,)
    assert plan.req_slice == (0, 1)
    assert plan.cu_seqlens == (0, 16)
    assert plan.start_pos == (8,)
    assert not plan.start_pos_zero
    assert plan.requires_history_state
    assert plan.compressed_row_indices == (0, 1, 2, 3)
    assert plan.compressed_row_slice == (0, 4)
    assert plan.output_keep_indices == (2,)
    assert plan.output_keep_slice == (2, 1)
    assert plan.slot_mapping_indices == (2,)
    assert plan.slot_mapping_slice == (2, 1)
    assert plan.local_keep_to_full_row_indices == (2,)


def test_c4_chunked_prefill_continuation_uses_history_plan():
    plan = _plan(
        ratio=4,
        positions=list(range(8192, 8196)),
        query_start_loc=[0, 4],
        seq_lens=[8196],
        local_start=0,
        local_end=4,
        is_chunked_prefill=True,
    )

    assert plan.enabled
    assert plan.is_chunked_prefill
    assert plan.start_pos == (8192,)
    assert plan.requires_history_state
    assert plan.requires_boundary_state_sync
    assert plan.global_compressed_row_count == 1
    assert plan.boundary_req_indices == (0,)
    assert plan.boundary_positions == (8195,)
    assert plan.boundary_owner_mask == (True,)
    assert plan.supports_boundary_state_replay
    assert plan.boundary_replay_token_ranges == ((0, 4),)
    assert plan.boundary_replay_token_slice == (0, 4)
    assert plan.boundary_replay_req_slice == (0, 1)
    assert plan.boundary_replay_cu_seqlens == (0, 4)
    assert plan.boundary_replay_start_pos == (8192,)
    assert plan.boundary_replay_compressed_row_slice == (0, 1)
    assert plan.boundary_replay_rope_row_indices == (0, 0)
    assert plan.boundary_replay_rope_row_slice is None
    assert plan.token_slice == (0, 4)
    assert plan.output_keep_slice == (0, 1)
    assert plan.local_keep_to_full_row_indices == (0,)


def test_c4_chunked_prefill_rank_without_output_uses_state_only_replay():
    plan = _plan(
        ratio=4,
        positions=list(range(8192, 8196)),
        query_start_loc=[0, 4],
        seq_lens=[8196],
        local_start=0,
        local_end=1,
        is_chunked_prefill=True,
        tp_size=8,
    )

    assert plan.enabled
    assert plan.reason == "enabled"
    assert plan.requires_boundary_state_sync
    assert plan.supports_boundary_state_replay
    assert plan.boundary_replay_token_slice == (0, 4)
    assert plan.token_slice == (0, 0)
    assert plan.output_keep_slice == (0, 0)
    assert plan.slot_mapping_slice == (0, 0)
    assert plan.local_keep_to_full_row_indices == ()


def test_c4_chunked_prefill_rank_without_tokens_uses_state_only_replay():
    plan = _plan(
        ratio=4,
        positions=list(range(8192, 8196)),
        query_start_loc=[0, 4],
        seq_lens=[8196],
        local_start=4,
        local_end=4,
        is_chunked_prefill=True,
        tp_size=8,
    )

    assert plan.enabled
    assert plan.supports_boundary_state_replay
    assert plan.token_slice == (0, 0)
    assert plan.output_keep_slice == (0, 0)


def test_c128_chunked_prefill_without_output_row_falls_back():
    plan = _plan(
        ratio=128,
        positions=list(range(8192, 8196)),
        query_start_loc=[0, 4],
        seq_lens=[8196],
        local_start=0,
        local_end=4,
        allow_c128_non_aligned=True,
        is_chunked_prefill=True,
    )

    assert not plan.enabled
    assert plan.is_chunked_prefill
    assert not plan.requires_boundary_state_sync
    assert plan.global_compressed_row_count == 0
    assert plan.boundary_req_indices == (0,)
    assert plan.boundary_positions == (8195,)
    assert plan.boundary_owner_mask == (True,)
    assert plan.reason == "no_local_compressed_rows"
    assert not plan.supports_boundary_state_replay


def test_c4_aligned_chunk_builds_state_only_boundary_replay():
    plan = _plan(
        ratio=4,
        positions=list(range(8192)),
        query_start_loc=[0, 8192],
        seq_lens=[8192],
        local_start=0,
        local_end=1024,
        is_chunked_prefill=True,
        tp_size=8,
    )

    assert plan.enabled
    assert plan.supports_boundary_state_replay
    assert plan.boundary_replay_token_ranges == ((8184, 8192),)
    assert plan.boundary_replay_token_slice == (8184, 8)
    assert plan.boundary_replay_req_slice == (0, 1)
    assert plan.boundary_replay_cu_seqlens == (0, 8)
    assert plan.boundary_replay_start_pos == (8184,)
    assert plan.boundary_replay_compressed_row_slice == (2046, 2)


def test_c4_nonzero_chunk_replay_adds_history_pad_and_aligns_to_state_row():
    plan = _plan(
        ratio=4,
        positions=list(range(64, 8256)),
        query_start_loc=[0, 8192],
        seq_lens=[8256],
        local_start=3072,
        local_end=4096,
        is_chunked_prefill=True,
        tp_size=8,
    )

    assert plan.enabled
    assert plan.supports_boundary_state_replay
    assert plan.boundary_replay_token_ranges == ((8176, 8192),)
    assert plan.boundary_replay_token_slice == (8176, 16)
    assert plan.boundary_replay_cu_seqlens == (0, 16)
    assert plan.boundary_replay_start_pos == (8240,)
    assert plan.boundary_replay_compressed_row_slice == (2044, 4)


def test_chunked_prefill_has_exactly_one_boundary_owner_across_tp_ranks():
    plans = [
        _plan(
            ratio=4,
            positions=list(range(8192)),
            query_start_loc=[0, 8192],
            seq_lens=[8192],
            local_start=rank * 1024,
            local_end=(rank + 1) * 1024,
            is_chunked_prefill=True,
            tp_size=8,
        )
        for rank in range(8)
    ]

    assert all(plan.requires_boundary_state_sync for plan in plans)
    assert all(plan.boundary_req_indices == (0,) for plan in plans)
    assert all(plan.boundary_positions == (8191,) for plan in plans)
    assert sum(plan.boundary_owner_mask[0] for plan in plans) == 1
    assert plans[7].boundary_owner_mask == (True,)
    assert all(plan.supports_boundary_state_replay for plan in plans)
    assert all(plan.boundary_replay_token_ranges == ((8184, 8192),) for plan in plans)


def test_chunked_prefill_boundary_owners_are_rank_invariant_for_packed_requests():
    positions = [0, 1, 2, *range(8), *range(9)]
    plans = [
        _plan(
            ratio=4,
            positions=positions,
            query_start_loc=[0, 3, 11, 20],
            seq_lens=[3, 8, 9],
            local_start=rank * 5,
            local_end=(rank + 1) * 5,
            allow_c4_non_aligned=True,
            is_chunked_prefill=True,
            tp_size=4,
        )
        for rank in range(4)
    ]

    assert all(plan.boundary_req_indices == (0, 1, 2) for plan in plans)
    assert all(plan.boundary_positions == (2, 7, 8) for plan in plans)
    assert all(plan.global_compressed_row_count == 4 for plan in plans)
    assert all(plan.supports_boundary_state_replay for plan in plans)
    assert all(plan.boundary_replay_token_ranges == ((0, 3), (3, 11), (11, 20)) for plan in plans)
    assert all(plan.boundary_replay_cu_seqlens == (0, 3, 11, 20) for plan in plans)
    assert all(plan.boundary_replay_compressed_row_indices == (0, 1, 2, 3) for plan in plans)
    for req_idx in range(3):
        assert sum(plan.boundary_owner_mask[req_idx] for plan in plans) == 1


def test_no_local_compressed_rows_falls_back():
    plan = _plan(
        ratio=128,
        positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=0,
        local_end=64,
    )

    assert not plan.enabled
    assert plan.reason == "no_local_compressed_rows"


def test_disabled_env_falls_back():
    plan = build_compressor_sp_plan(
        enabled=False,
        has_prefill=True,
        need_gather_q_kv=True,
        tp_size=4,
        compress_ratio=4,
        input_positions=list(range(8)),
        query_start_loc=[0, 8],
        seq_lens=[8],
        local_start=0,
        local_end=4,
    )

    assert not plan.enabled
    assert plan.reason == "env_disabled"


# --- Tests for sp_row_counts_per_rank, tp_rank, tp_size ---


def test_sp_row_counts_c4_even_split():
    """C4 with 16 tokens, tp_size=4: tokens_per_rank=4, each rank owns 4 tokens.
    Compressed rows at flat 3, 7, 11, 15 => one per rank."""
    plan = _plan(
        ratio=4,
        positions=list(range(16)),
        query_start_loc=[0, 16],
        seq_lens=[16],
        local_start=0,
        local_end=4,
        tp_size=4,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 1, 1, 1)
    assert plan.gather_compact_indices == (0, 1, 2, 3)
    assert plan.gather_compact_slice == (0, 4)
    assert plan.tp_rank == 0
    assert plan.tp_size == 4
    assert plan.global_compressed_row_count == 4
    assert sum(plan.sp_row_counts_per_rank) == plan.global_compressed_row_count


def test_sp_row_counts_c4_uneven_rows():
    """C4 with 32 tokens, tp_size=4: tokens_per_rank=8.
    Compressed rows at positions 3,7,11,15,19,23,27,31.
    Rank 0 owns [0,8) => flat 3,7 => 2 rows.
    Rank 1 owns [8,16) => flat 11,15 => 2 rows.
    Rank 2 owns [16,24) => flat 19,23 => 2 rows.
    Rank 3 owns [24,32) => flat 27,31 => 2 rows."""
    plan = _plan(
        ratio=4,
        positions=list(range(32)),
        query_start_loc=[0, 32],
        seq_lens=[32],
        local_start=0,
        local_end=8,
        tp_size=4,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (2, 2, 2, 2)
    assert plan.global_compressed_row_count == 8
    assert sum(plan.sp_row_counts_per_rank) == plan.global_compressed_row_count


def test_gather_compact_selector_skips_padded_rank_rows():
    """C4/20 tokens over TP3 owns (1, 2, 2) rows with max_rows=2."""
    plan = _plan(
        ratio=4,
        positions=list(range(20)),
        query_start_loc=[0, 20],
        seq_lens=[20],
        local_start=0,
        local_end=7,
        tp_size=3,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 2, 2)
    assert plan.gather_compact_indices == (0, 2, 3, 4, 5)
    assert plan.gather_compact_slice is None


def test_sp_row_counts_c128_two_ranks():
    """C128 with 256 tokens, tp_size=2: tokens_per_rank=128.
    Compressed rows at flat 127 and 255.
    Rank 0 owns [0,128) => row at 127 => 1 row.
    Rank 1 owns [128,256) => row at 255 => 1 row."""
    plan = _plan(
        ratio=128,
        positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=0,
        local_end=128,
        tp_size=2,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 1)
    assert plan.tp_rank == 0
    assert plan.tp_size == 2


def test_sp_row_counts_c4_multiple_rows_per_rank():
    """C4 with 32 tokens, tp_size=2: tokens_per_rank=16.
    Compressed rows at flat 3,7,11,15,19,23,27,31 => 8 rows total.
    Rank 0 owns [0,16) => rows at flat 3,7,11,15 => 4 rows.
    Rank 1 owns [16,32) => rows at flat 19,23,27,31 => 4 rows."""
    plan = _plan(
        ratio=4,
        positions=list(range(32)),
        query_start_loc=[0, 32],
        seq_lens=[32],
        local_start=0,
        local_end=16,
        tp_size=2,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (4, 4)
    assert plan.global_compressed_row_count == 8


def test_sp_row_counts_preserved_in_tp_rank_field():
    """Verify that tp_rank is correctly preserved for different ranks."""
    positions = list(range(16))
    for rank in range(4):
        plan = _plan(
            ratio=4,
            positions=positions,
            query_start_loc=[0, 16],
            seq_lens=[16],
            local_start=rank * 4,
            local_end=(rank + 1) * 4,
            tp_size=4,
            tp_rank=rank,
        )
        assert plan.enabled
        assert plan.tp_rank == rank
        assert plan.tp_size == 4
        # All ranks compute the same sp_row_counts_per_rank
        assert plan.sp_row_counts_per_rank == (1, 1, 1, 1)


def test_sp_row_counts_multi_request():
    """Multiple requests: 8 tokens each, total 16, tp_size=2.
    positions: [0..7] + [0..7], compressed rows at flat 3,7,11,15.
    tokens_per_rank = 8.
    Rank 0 owns [0,8) => flat 3,7 => 2 rows.
    Rank 1 owns [8,16) => flat 11,15 => 2 rows."""
    plan = _plan(
        ratio=4,
        positions=list(range(8)) + list(range(8)),
        query_start_loc=[0, 8, 16],
        seq_lens=[8, 8],
        local_start=0,
        local_end=8,
        tp_size=2,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (2, 2)
    assert plan.global_compressed_row_count == 4


def test_sp_row_counts_zero_rows_rank():
    """C128 with 256 tokens, tp_size=2: tokens_per_rank=128.
    Compressed rows at flat 127, 255.
    Rank 0 owns [0,128) => row at 127 => 1 row.
    Rank 1 owns [128,256) => row at 255 => 1 row.
    Both ranks produce the same sp_row_counts_per_rank since it's
    computed from global input_positions, not just local data."""
    plan = _plan(
        ratio=128,
        positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=128,
        local_end=256,
        tp_size=2,
        tp_rank=1,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 1)
    assert plan.tp_rank == 1


def test_sp_row_counts_sum_equals_global_count():
    """Verify that sp_row_counts always sums to global_compressed_row_count
    across various configurations."""
    configs = [
        # (ratio, num_tokens, tp_size)
        (4, 16, 4),
        (4, 32, 2),
        (4, 64, 8),
        (128, 256, 2),
        (128, 512, 4),
    ]
    for ratio, num_tokens, tp_size in configs:
        positions = list(range(num_tokens))
        tokens_per_rank = (((num_tokens + tp_size - 1) // tp_size) * tp_size) // tp_size
        plan = _plan(
            ratio=ratio,
            positions=positions,
            query_start_loc=[0, num_tokens],
            seq_lens=[num_tokens],
            local_start=0,
            local_end=tokens_per_rank,
            tp_size=tp_size,
            tp_rank=0,
        )
        if plan.enabled:
            assert sum(plan.sp_row_counts_per_rank) == plan.global_compressed_row_count, (
                f"Failed for ratio={ratio}, num_tokens={num_tokens}, tp_size={tp_size}: "
                f"sum({plan.sp_row_counts_per_rank}) != {plan.global_compressed_row_count}"
            )


def test_zero_row_rank_layout_is_detected_before_collective():
    plan = _plan(
        ratio=4,
        positions=list(range(8)),
        query_start_loc=[0, 8],
        seq_lens=[8],
        local_start=2,
        local_end=4,
        tp_size=4,
        tp_rank=1,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (0, 1, 0, 1)
    assert plan.gather_compact_indices == (1, 3)
    assert not all_ranks_have_compressor_sp_rows(plan.sp_row_counts_per_rank)
