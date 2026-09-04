import pytest
import torch

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_padded_destination_for_scatter,
    flatten_slot_mapping,
    fused_gather_rows,
    is_block_offset_slot_mapping,
    select_block_cache_rows,
    update_block_cache_rows_,
)


def _fake_all_gather(per_rank_locals):
    """Simulate ``all_gather_into_tensor`` without a process group.

    ``per_rank_locals`` maps this rank's flat local buffer to the contribution of
    every rank, so the packing and unpacking layout can be checked on CPU.
    """
    calls = []

    def all_gather_into_tensor(gathered, local):
        calls.append(local.clone())
        blocks = per_rank_locals(local)
        assert gathered.numel() == sum(block.numel() for block in blocks)
        offset = 0
        for block in blocks:
            gathered[offset : offset + block.numel()] = block.reshape(-1)
            offset += block.numel()

    return all_gather_into_tensor, calls


def test_fused_gather_issues_one_collective_and_rebuilds_rank_major_blocks():
    tp_size = 3
    out_counts = (2, 1, 2)  # ragged: max_rows 2, so rank 1 contributes a pad row
    state_counts = (1, 1, 1)
    local_out = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    local_state = torch.tensor([[10.0, 20.0, 30.0]])

    all_gather, calls = _fake_all_gather(lambda local: [local + rank * 100.0 for rank in range(tp_size)])
    result = fused_gather_rows(
        ((local_out, out_counts), (local_state, state_counts)),
        tp_size,
        all_gather,
    )

    assert result is not None
    assert len(calls) == 1, "both blocks must share a single collective"
    # One flat buffer per rank: 2 padded output rows x 2 + 1 state row x 3.
    assert calls[0].numel() == 2 * 2 + 1 * 3
    assert calls[0].tolist() == [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]

    gathered_out, gathered_state = result
    assert gathered_out.shape == (tp_size * 2, 2)
    assert gathered_state.shape == (tp_size * 1, 3)
    # Rank-major order, matching the per-block gather layout the selectors index.
    assert gathered_out.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [101.0, 102.0],
        [103.0, 104.0],
        [201.0, 202.0],
        [203.0, 204.0],
    ]
    assert gathered_state.tolist() == [
        [10.0, 20.0, 30.0],
        [110.0, 120.0, 130.0],
        [210.0, 220.0, 230.0],
    ]


def test_fused_gather_matches_two_separate_gathers():
    """The fused layout must equal what per-block gathers produce."""
    tp_size = 4
    out_counts = (3, 3, 2, 3)
    state_counts = (2, 1, 2, 2)
    local_out = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5)
    local_state = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 50.0

    all_gather, _ = _fake_all_gather(lambda local: [local + rank * 1000.0 for rank in range(tp_size)])
    fused = fused_gather_rows(
        ((local_out, out_counts), (local_state, state_counts)),
        tp_size,
        all_gather,
    )
    assert fused is not None

    def reference(local_rows, row_counts):
        max_rows = max(row_counts)
        padded = local_rows.new_zeros((max_rows, *local_rows.shape[1:]))
        padded[: local_rows.shape[0]] = local_rows
        return torch.cat([padded + rank * 1000.0 for rank in range(tp_size)], dim=0)

    assert fused[0].equal(reference(local_out, out_counts))
    assert fused[1].equal(reference(local_state, state_counts))


def test_fused_gather_zero_pads_short_blocks():
    """A rank owning fewer rows than the maximum must send zeros, not garbage."""
    tp_size = 2
    local_out = torch.tensor([[7.0]])  # owns 1 row, planned maximum is 3
    all_gather, _ = _fake_all_gather(lambda local: [local, local])

    fused = fused_gather_rows(((local_out, (1, 3)),), tp_size, all_gather)

    assert fused is not None
    assert fused[0].shape == (tp_size * 3, 1)
    assert fused[0][:, 0].tolist() == [7.0, 0.0, 0.0, 7.0, 0.0, 0.0]


def test_fused_gather_declines_mixed_dtypes():
    """Mixed dtypes cannot share a buffer, and declining must not send anything."""
    all_gather, calls = _fake_all_gather(lambda local: [local])
    result = fused_gather_rows(
        (
            (torch.zeros((1, 2), dtype=torch.float32), (1,)),
            (torch.zeros((1, 2), dtype=torch.float16), (1,)),
        ),
        1,
        all_gather,
    )

    assert result is None
    assert calls == []


def test_fused_gather_declines_empty_and_oversized_payloads():
    all_gather, calls = _fake_all_gather(lambda local: [local])

    assert fused_gather_rows((), 2, all_gather) is None
    # Nothing to send: every planned block is empty.
    assert fused_gather_rows(((torch.zeros((0, 4)), (0, 0)),), 2, all_gather) is None
    # A block longer than its planned per-rank maximum would break the layout.
    assert fused_gather_rows(((torch.zeros((3, 4)), (1, 2)),), 2, all_gather) is None
    assert calls == []


def _linear_slot(slot, block_size):
    return int(slot[0]) * block_size + int(slot[1])


def test_ragged_padded_destination_matches_compact_mapping():
    block_size = 128
    full_slot_mapping = torch.tensor(
        [[10, 1], [11, 2], [12, 3], [13, 4], [14, 5]],
        dtype=torch.int64,
    )
    gather_indices = torch.tensor([0, 2, 3, 4, 5])

    padded, _ = build_padded_destination_for_scatter(
        full_slot_mapping,
        gather_indices,
        padded_rows=6,
        block_size=block_size,
    )

    assert padded.index_select(0, gather_indices).equal(full_slot_mapping)
    assert padded[1].tolist() == [-1, block_size - 1]
    assert _linear_slot(padded[1], block_size) == -1


def test_reused_buffer_resets_stale_padding_rows():
    block_size = 128
    first, buffer = build_padded_destination_for_scatter(
        torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0]]),
        torch.tensor([0, 1, 2, 3]),
        padded_rows=4,
        block_size=block_size,
    )
    assert first.shape == (4, 2)

    second, reused = build_padded_destination_for_scatter(
        torch.tensor([[4, 0], [5, 0]]),
        torch.tensor([0, 2]),
        padded_rows=4,
        block_size=block_size,
        buffer=buffer,
    )

    assert reused.data_ptr() == buffer.data_ptr()
    assert second.tolist() == [[4, 0], [-1, 127], [5, 0], [-1, 127]]


def test_a5_flat_and_non_a5_block_offset_mapping_detection():
    assert is_block_offset_slot_mapping(torch.zeros((4, 2), dtype=torch.int32))
    assert not is_block_offset_slot_mapping(torch.zeros(4, dtype=torch.int32))
    assert not is_block_offset_slot_mapping(torch.zeros((4, 1), dtype=torch.int32))


def test_state_slot_mapping_flattens_both_device_formats():
    block_offset = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32)
    flat = torch.tensor([259, 517], dtype=torch.int32)

    assert flatten_slot_mapping(block_offset, 128).tolist() == [259, 517]
    assert flatten_slot_mapping(flat, 128).tolist() == [259, 517]


def test_invalid_state_slot_mapping_shape_is_rejected():
    with pytest.raises(ValueError, match="slot mapping"):
        flatten_slot_mapping(torch.zeros((2, 1), dtype=torch.int32), 128)


def test_state_cache_rows_support_non_contiguous_block_storage():
    storage = torch.zeros((2, 5, 4), dtype=torch.float32)
    cache_rows = storage[:, :3, :]
    assert not cache_rows.is_contiguous()
    flat_slots = torch.tensor([1, 4], dtype=torch.int32)
    updates = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    update_block_cache_rows_(cache_rows, flat_slots, updates, block_size=3)

    assert select_block_cache_rows(cache_rows, flat_slots, block_size=3).equal(updates)
    assert storage[0, 1].equal(updates[0])
    assert storage[1, 1].equal(updates[1])
