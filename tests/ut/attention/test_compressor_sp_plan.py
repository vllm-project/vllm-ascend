import pytest

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_compressor_sp_plan,
)


def _plan(**overrides):
    params = dict(
        enabled=True,
        has_prefill=True,
        has_decode=False,
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


def _selector_values(plan, name):
    selector_slice = getattr(plan, f"{name}_slice")
    if selector_slice is not None:
        start, length = selector_slice
        return tuple(range(start, start + length))
    return getattr(plan, f"{name}_indices")


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
    assert plan.output_keep_indices == ()
    assert plan.output_keep_slice == (2, 512)
    assert plan.token_slice is not None
    assert plan.token_slice[1] >= 2048
    assert plan.req_indices == (0,)
    assert plan.state_replay_token_slice == (8184, 8)
    assert plan.state_replay_req_slice == (0, 1)
    assert plan.state_replay_cu_seqlens == (0, 8)
    assert plan.state_replay_start_pos == (8184,)
    assert plan.state_replay_rope_row_indices == ()
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
    assert plan.output_keep_indices == ()
    assert plan.output_keep_slice == (0, 1)
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
    assert max(_selector_values(plan, "token")) < 8
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


def test_mixed_decode_prefill_falls_back_before_planning():
    plan = _plan(has_decode=True)

    assert not plan.enabled
    assert plan.reason == "mixed_decode_prefill"


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
    assert plan.gather_compact_indices == ()
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
    assert plan.output_keep_indices == ()
    assert plan.output_keep_slice == (2, 2)
    assert not plan.requires_state_sync
    assert plan.state_sync_row_counts_per_rank == (0, 0)
    assert plan.state_sync_global_token_indices == ()
    assert plan.state_replay_token_slice == (0, 16)
    assert plan.state_replay_cu_seqlens == (0, 16)
    assert plan.state_replay_start_pos == (128,)
    assert plan.state_replay_rope_row_indices == ()
    assert plan.state_replay_rope_row_slice == (0, 5)


def test_c4_continuing_chunk_replays_boundary_state():
    plan = _plan(
        input_positions=list(range(8192, 16384)),
        query_start_loc=[0, 8192],
        seq_lens=[16384],
        local_start=6144,
        local_end=8192,
        tp_rank=3,
    )

    assert plan.enabled
    assert plan.token_slice == (6136, 2056)
    assert plan.output_keep_slice == (2, 512)
    assert plan.gather_compact_slice == (0, 2048)
    assert not plan.requires_state_sync
    assert plan.state_replay_token_slice == (8176, 16)
    assert plan.state_replay_start_pos == (16368,)


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
    assert plan.rope_row_indices == ()
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
    assert max(_selector_values(plan, "token")) < 8


def _ref_sp_row_counts(num_tokens, ratio, tp_size):
    """Per-token reference for the closed-form owned-row bucketing."""
    num_tokens_pad = ((num_tokens + tp_size - 1) // tp_size) * tp_size
    tokens_per_rank = num_tokens_pad // tp_size
    counts = [0] * tp_size
    for flat_index in range(num_tokens):
        if (flat_index + 1) % ratio == 0:
            counts[min(flat_index // tokens_per_rank, tp_size - 1)] += 1
    return tuple(counts)


@pytest.mark.parametrize("num_tokens", [4096, 4097, 8192, 8193, 12288])
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_closed_form_owned_counts_match_per_token_reference(num_tokens, ratio, tp_size):
    """The fixed-owner planner must match the per-token walk."""
    tokens_per_rank = (num_tokens + tp_size - 1) // tp_size
    reference = _ref_sp_row_counts(num_tokens, ratio, tp_size)
    for tp_rank in range(tp_size):
        local_start = tp_rank * tokens_per_rank
        local_end = local_start + tokens_per_rank
        plan = _plan(
            tp_size=tp_size,
            compress_ratio=ratio,
            input_positions=list(range(num_tokens)),
            query_start_loc=[0, num_tokens],
            seq_lens=[num_tokens],
            local_start=local_start,
            local_end=local_end,
            tp_rank=tp_rank,
        )
        assert plan.enabled, plan.reason
        assert plan.sp_row_counts_per_rank == reference
        # gather_compact must reindex the padded rank-major buffer back to the
        # dense global row order (0..sum(counts)-1) exactly once.
        gather_compact = _selector_values(plan, "gather_compact")
        assert sorted(gather_compact) == sorted(set(gather_compact))
        assert len(gather_compact) == sum(reference)


def test_endpoint_position_check_relies_on_vllm_contiguity_invariant():
    positions = list(range(16))
    positions[7] = 99  # interior corruption, endpoints still consistent

    plan = _plan(input_positions=positions)
    assert plan.enabled  # endpoints (0 and 15) still line up


def _reference_plan_selectors(*, num_tokens, ratio, tp_size, tp_rank, query_start_loc, seq_lens, positions):
    """Per-token reference for the arithmetic output-row passes.

    Mirrors the original scan: walk every token, emit a global compressed row
    wherever ``(position + 1) % ratio == 0``, bucket it by owning rank, and build
    the rank-major gather layout from those buckets.
    """
    tokens_per_rank = (((num_tokens + tp_size - 1) // tp_size) * tp_size) // tp_size
    num_reqs = len(query_start_loc) - 1
    query_lens = [query_start_loc[i + 1] - query_start_loc[i] for i in range(num_reqs)]
    request_starts = [seq_lens[i] - query_lens[i] for i in range(num_reqs)]

    owned_rows = [[] for _ in range(tp_size)]
    flat_to_row = {}
    row = 0
    for flat_index, position in enumerate(positions):
        if (position + 1) % ratio == 0:
            flat_to_row[flat_index] = row
            owned_rows[min(flat_index // tokens_per_rank, tp_size - 1)].append(row)
            row += 1

    max_rows = max(len(rows) for rows in owned_rows)
    gather = [0] * row
    for rank, rows in enumerate(owned_rows):
        for local_row, global_row in enumerate(rows):
            gather[global_row] = rank * max_rows + local_row

    local_start = tp_rank * tokens_per_rank
    local_end = min(local_start + tokens_per_rank, num_tokens)
    owned_flat = [
        flat_index
        for req_index in range(num_reqs)
        for flat_index in range(
            max(query_start_loc[req_index], local_start),
            min(query_start_loc[req_index + 1], local_end),
        )
    ]
    keep_rows = [flat_to_row[flat_index] for flat_index in owned_flat if flat_index in flat_to_row]
    return {
        "sp_row_counts_per_rank": tuple(len(rows) for rows in owned_rows),
        "gather_compact_indices": tuple(gather),
        "num_keep_rows": len(keep_rows),
        "request_starts": request_starts,
    }


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("tp_size", [2, 3, 4, 8])
@pytest.mark.parametrize(("num_tokens", "chunk_start"), [(4096, 0), (8192, 0), (8192, 8192), (5120, 4096), (60, 0)])
def test_arithmetic_output_rows_match_per_token_reference(ratio, tp_size, num_tokens, chunk_start):
    """Every rank's selectors must equal the per-token scan they replaced."""
    positions = list(range(chunk_start, chunk_start + num_tokens))
    query_start_loc = [0, num_tokens]
    seq_lens = [chunk_start + num_tokens]
    tokens_per_rank = (((num_tokens + tp_size - 1) // tp_size) * tp_size) // tp_size

    for tp_rank in range(tp_size):
        plan = _plan(
            tp_size=tp_size,
            compress_ratio=ratio,
            input_positions=positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            local_start=tp_rank * tokens_per_rank,
            local_end=min((tp_rank + 1) * tokens_per_rank, num_tokens),
            tp_rank=tp_rank,
        )
        reference = _reference_plan_selectors(
            num_tokens=num_tokens,
            ratio=ratio,
            tp_size=tp_size,
            tp_rank=tp_rank,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            positions=positions,
        )
        assert plan.enabled, plan.reason
        assert plan.sp_row_counts_per_rank == reference["sp_row_counts_per_rank"]
        assert _selector_values(plan, "gather_compact") == reference["gather_compact_indices"]
        assert len(_selector_values(plan, "output_keep")) == reference["num_keep_rows"]


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("lens", [[2048, 2048], [1024, 2048, 1024], [16, 16]])
def test_packed_requests_keep_per_request_output_phase(ratio, lens):
    """Each request restarts the output phase at its own start position."""
    num_tokens = sum(lens)
    positions = [position for length in lens for position in range(length)]
    query_start_loc = [0]
    for length in lens:
        query_start_loc.append(query_start_loc[-1] + length)
    tp_size = 2
    tokens_per_rank = (((num_tokens + tp_size - 1) // tp_size) * tp_size) // tp_size

    for tp_rank in range(tp_size):
        plan = _plan(
            tp_size=tp_size,
            compress_ratio=ratio,
            input_positions=positions,
            query_start_loc=query_start_loc,
            seq_lens=list(lens),
            local_start=tp_rank * tokens_per_rank,
            local_end=min((tp_rank + 1) * tokens_per_rank, num_tokens),
            tp_rank=tp_rank,
        )
        reference = _reference_plan_selectors(
            num_tokens=num_tokens,
            ratio=ratio,
            tp_size=tp_size,
            tp_rank=tp_rank,
            query_start_loc=query_start_loc,
            seq_lens=list(lens),
            positions=positions,
        )
        assert plan.enabled, plan.reason
        assert plan.sp_row_counts_per_rank == reference["sp_row_counts_per_rank"]
        assert _selector_values(plan, "gather_compact") == reference["gather_compact_indices"]


def test_contiguous_slice_rejects_permuted_selectors():
    """A ragged gather selector must not be collapsed into a narrow() slice."""
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
    # Same first/last value as range(0, 5) but permuted in the middle.
    assert plan.gather_compact_indices == (0, 2, 3, 4, 5)
    assert plan.gather_compact_slice is None
