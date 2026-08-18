import pytest
import torch

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_padded_destination_for_scatter,
    is_block_offset_slot_mapping,
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


def test_a5_flat_slot_mapping_is_rejected_by_block_offset_helper():
    with pytest.raises(ValueError, match="block-offset"):
        build_padded_destination_for_scatter(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            padded_rows=2,
            block_size=128,
        )
