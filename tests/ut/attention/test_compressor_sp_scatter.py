import pytest
import torch

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_padded_destination_for_scatter,
    flatten_slot_mapping,
    is_block_offset_slot_mapping,
    select_block_cache_rows,
    update_block_cache_rows_,
)


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
