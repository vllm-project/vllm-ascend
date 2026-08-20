import pytest

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_compressor_sp_plan,
)


def _plan(**overrides):
    params = dict(
        enabled=True,
        has_prefill=True,
        need_gather_q_kv=True,
        tp_size=4,
        compress_ratio=4,
        input_positions=list(range(16)),
        query_start_loc=[0, 16],
        seq_lens=[16],
        local_start=0,
        local_end=4,
        tp_rank=0,
    )
    params.update(overrides)
    return build_compressor_sp_plan(**params)


def test_c4_aligned_8192_plan_uses_contiguous_gather():
    plan = _plan(
        input_positions=list(range(8192)),
        query_start_loc=[0, 8192],
        seq_lens=[8192],
        local_start=2048,
        local_end=4096,
        tp_rank=1,
    )

    assert plan.enabled
    assert plan.num_input_tokens == 8192
    assert plan.sp_row_counts_per_rank == (512, 512, 512, 512)
    assert plan.gather_compact_slice == (0, 2048)
    assert len(plan.output_keep_indices) == 512
    assert plan.token_slice is not None
    assert plan.token_slice[1] >= 2048
    assert plan.req_indices == (0,)
    assert plan.state_replay_token_slice == (8184, 8)
    assert plan.state_replay_req_slice == (0, 1)
    assert plan.state_replay_cu_seqlens == (0, 8)
    assert plan.state_replay_start_pos == (8184,)
    assert plan.state_replay_rope_row_indices == (2046, 2047, 2048)
    assert plan.state_replay_rope_row_slice == (2046, 3)


@pytest.mark.parametrize(
    ("rank", "local_start", "local_end", "expected_start"),
    [(0, 0, 128, 0), (1, 128, 256, 128)],
)
def test_c128_aligned_rank_mapping(rank, local_start, local_end, expected_start):
    plan = _plan(
        tp_size=2,
        compress_ratio=128,
        input_positions=list(range(256)),
        query_start_loc=[0, 256],
        seq_lens=[256],
        local_start=local_start,
        local_end=local_end,
        tp_rank=rank,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 1)
    assert plan.token_slice == (expected_start, 128)
    assert plan.output_keep_indices == (0,)
    assert plan.gather_compact_slice == (0, 2)
    assert not plan.requires_state_sync
    assert plan.state_sync_row_counts_per_rank == (0, 0)
    assert plan.state_sync_global_token_indices == ()


def test_multi_request_overlap_does_not_cross_request_boundary():
    positions = list(range(8)) + list(range(8))
    plan = _plan(
        tp_size=2,
        input_positions=positions,
        query_start_loc=[0, 8, 16],
        seq_lens=[8, 8],
        local_start=0,
        local_end=8,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.req_indices == (0,)
    assert plan.token_indices
    assert max(plan.token_indices) < 8
    assert plan.sp_row_counts_per_rank == (2, 2)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"enabled": False}, "disabled"),
        ({"has_prefill": False}, "unsupported_attention_state"),
        ({"need_gather_q_kv": False}, "no_sequence_parallelism"),
        ({"compress_ratio": 8}, "unsupported_ratio"),
        (
            {
                "input_positions": list(range(4)),
                "query_start_loc": [0, 4],
                "seq_lens": [3],
                "local_end": 1,
            },
            "negative_start_pos",
        ),
        (
            {
                "input_positions": list(range(15)) + [17],
            },
            "noncontiguous_positions",
        ),
        (
            {
                "input_positions": list(range(5)),
                "query_start_loc": [0, 5],
                "seq_lens": [5],
            },
            "zero_token_rank",
        ),
        ({"query_start_loc": [0, 17]}, "query_start_loc_out_of_bounds"),
    ],
)
def test_unsupported_shapes_fall_back(overrides, reason):
    plan = _plan(**overrides)
    assert not plan.enabled
    assert plan.reason == reason


def test_adaptive_gate_rejects_small_chunks_before_planning():
    plan = _plan(min_input_tokens=32)

    assert not plan.enabled
    assert plan.reason == "adaptive_small_chunk"


def test_zero_output_row_rank_still_participates_in_collective():
    plan = _plan(
        input_positions=list(range(8)),
        query_start_loc=[0, 8],
        seq_lens=[8],
        local_start=0,
        local_end=2,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (0, 1, 0, 1)
    assert plan.output_keep_indices == ()
    assert not plan.requires_state_sync
    assert plan.state_sync_row_counts_per_rank == (0, 0, 0, 0)
    assert plan.state_sync_token_indices == ()


def test_ragged_gather_selector_skips_padding_rows():
    plan = _plan(
        tp_size=3,
        input_positions=list(range(20)),
        query_start_loc=[0, 20],
        seq_lens=[20],
        local_start=0,
        local_end=7,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (1, 2, 2)
    assert plan.gather_compact_indices == (0, 2, 3, 4, 5)
    assert plan.gather_compact_slice is None


def test_ragged_trailing_padding_keeps_physical_indices():
    plan = _plan(
        tp_size=8,
        input_positions=list(range(60)),
        query_start_loc=[0, 60],
        seq_lens=[60],
        local_start=0,
        local_end=8,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (2, 2, 2, 2, 2, 2, 2, 1)
    assert plan.gather_compact_indices == tuple(range(15))
    assert plan.gather_compact_slice == (0, 15)


def test_c4_chunked_prefill_uses_absolute_start_position():
    plan = _plan(
        tp_size=2,
        input_positions=list(range(128, 144)),
        query_start_loc=[0, 16],
        seq_lens=[144],
        local_start=8,
        local_end=16,
        tp_rank=1,
    )

    assert plan.enabled
    assert plan.start_pos == (128,)
    assert plan.sp_row_counts_per_rank == (2, 2)
    assert plan.output_keep_indices == (2, 3)
    assert not plan.requires_state_sync
    assert plan.state_sync_row_counts_per_rank == (0, 0)
    assert plan.state_sync_global_token_indices == ()
    assert plan.state_replay_token_slice == (0, 16)
    assert plan.state_replay_cu_seqlens == (0, 16)
    assert plan.state_replay_start_pos == (128,)
    assert plan.state_replay_rope_row_indices == (0, 1, 2, 3, 4)
    assert plan.state_replay_rope_row_slice == (0, 5)


def test_c4_multi_request_replay_uses_full_batch_rope_shape():
    plan = _plan(
        tp_size=2,
        input_positions=list(range(128, 143)) + [256],
        query_start_loc=[0, 15, 16],
        seq_lens=[143, 257],
        local_start=0,
        local_end=8,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.state_replay_token_slice == (0, 16)
    assert plan.state_replay_req_slice == (0, 2)
    assert plan.state_replay_cu_seqlens == (0, 15, 16)
    assert len(plan.state_replay_rope_row_indices) == 6
    assert plan.state_replay_rope_row_indices == (0, 1, 2, 0, 0, 0)


def test_rope_padding_falls_back_when_no_contiguous_source_row_exists():
    plan = _plan(
        tp_size=2,
        input_positions=list(range(4)),
        query_start_loc=[0, 4],
        seq_lens=[4],
        local_start=0,
        local_end=2,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.rope_row_indices == (0,)
    assert plan.rope_row_slice == (0, 1)


def test_c128_chunked_prefill_tracks_ragged_state_tail():
    plan = _plan(
        tp_size=2,
        compress_ratio=128,
        input_positions=list(range(126, 258)),
        query_start_loc=[0, 132],
        seq_lens=[258],
        local_start=66,
        local_end=132,
        tp_rank=1,
    )

    assert plan.enabled
    assert plan.start_pos == (128,)
    assert plan.sp_row_counts_per_rank == (1, 1)
    assert plan.requires_state_sync
    assert plan.state_sync_row_counts_per_rank == (0, 2)
    assert plan.state_sync_global_token_indices == (130, 131)
    assert plan.state_sync_gather_compact_slice == (2, 2)
    assert plan.state_replay_token_indices == ()
    assert plan.state_replay_cu_seqlens == ()


def test_chunked_prefill_can_update_state_without_compressed_output():
    plan = _plan(
        tp_size=2,
        compress_ratio=128,
        input_positions=[128, 129],
        query_start_loc=[0, 2],
        seq_lens=[130],
        local_start=0,
        local_end=1,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.sp_row_counts_per_rank == (0, 0)
    assert plan.output_keep_indices == ()
    assert plan.start_pos == (128,)
    assert plan.requires_state_sync
    assert plan.state_sync_global_token_indices == (0, 1)


def test_chunked_multi_request_positions_are_validated_per_request():
    plan = _plan(
        tp_size=2,
        input_positions=list(range(124, 132)) + list(range(252, 260)),
        query_start_loc=[0, 8, 16],
        seq_lens=[132, 260],
        local_start=0,
        local_end=8,
        tp_rank=0,
    )

    assert plan.enabled
    assert plan.req_indices == (0,)
    assert plan.start_pos == (124,)
    assert plan.sp_row_counts_per_rank == (2, 2)
    assert max(plan.token_indices) < 8
