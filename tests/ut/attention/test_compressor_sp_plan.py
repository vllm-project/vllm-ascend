import pytest

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_compressor_sp_plan,
)


def _plan(**overrides):
    params = dict(
        enabled=True,
        is_full_prefill=True,
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
    assert plan.sp_row_counts_per_rank == (512, 512, 512, 512)
    assert plan.gather_compact_slice == (0, 2048)
    assert len(plan.output_keep_indices) == 512
    assert plan.token_slice is not None
    assert plan.token_slice[1] >= 2048
    assert plan.req_indices == (0,)


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
        ({"is_full_prefill": False}, "unsupported_attention_state"),
        ({"need_gather_q_kv": False}, "no_sequence_parallelism"),
        ({"compress_ratio": 8}, "unsupported_ratio"),
        (
            {
                "input_positions": list(range(4)),
                "query_start_loc": [0, 4],
                "seq_lens": [8],
                "local_end": 1,
            },
            "nonzero_start_pos",
        ),
        (
            {
                "input_positions": list(range(5)),
                "query_start_loc": [0, 5],
                "seq_lens": [5],
                "local_end": 2,
            },
            "seq_len_not_aligned",
        ),
        ({"query_start_loc": [0, 17]}, "query_start_loc_out_of_bounds"),
    ],
)
def test_unsupported_shapes_fall_back(overrides, reason):
    plan = _plan(**overrides)
    assert not plan.enabled
    assert plan.reason == reason


def test_zero_row_rank_is_rejected_before_collective():
    plan = _plan(
        input_positions=list(range(8)),
        query_start_loc=[0, 8],
        seq_lens=[8],
        local_start=2,
        local_end=4,
        tp_rank=1,
    )

    assert not plan.enabled
    assert plan.reason == "zero_row_rank"


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
    assert plan.gather_compact_slice is None
