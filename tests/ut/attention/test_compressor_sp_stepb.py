"""CPU tests for the Step B padded-destination production helper."""

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_padded_destination_for_scatter,
)


def linear_slot(slot_row, block_size):
    """Mirror the block-offset linearization used by the Scatter kernel."""
    block_id, block_offset = slot_row
    return int(block_id) * block_size + int(block_offset)


def build_padded_destination(full_slot_mapping, gather_compact_indices, max_rows, tp_size, block_size):
    """Call the production helper while retaining the NumPy golden checks."""
    padded, _ = build_padded_destination_for_scatter(
        torch.as_tensor(full_slot_mapping),
        torch.as_tensor(gather_compact_indices),
        None,
        max_rows,
        tp_size,
        block_size,
    )
    return padded.numpy()


def test_reused_buffer_resets_stale_padding_rows():
    block_size = 128
    first, buffer = build_padded_destination_for_scatter(
        torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0]]),
        torch.tensor([0, 1, 2, 3]),
        None,
        max_rows=2,
        tp_size=2,
        block_size=block_size,
    )
    assert first.shape == (4, 2)

    second, reused = build_padded_destination_for_scatter(
        torch.tensor([[4, 0], [5, 0]]),
        torch.tensor([0, 2]),
        None,
        max_rows=2,
        tp_size=2,
        block_size=block_size,
        buffer=buffer,
    )
    assert reused.data_ptr() == buffer.data_ptr()
    assert second.tolist() == [[4, 0], [-1, 127], [5, 0], [-1, 127]]


def test_contiguous_selector_copies_requested_rows():
    padded, _ = build_padded_destination_for_scatter(
        torch.tensor([[10, 1], [11, 2], [12, 3], [13, 4]]),
        None,
        (1, 2),
        max_rows=1,
        tp_size=2,
        block_size=128,
    )
    assert padded.tolist() == [[11, 2], [12, 3]]


def test_flat_slot_mapping_is_rejected():
    with pytest.raises(ValueError, match="block-offset"):
        build_padded_destination_for_scatter(
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            None,
            max_rows=1,
            tp_size=2,
            block_size=128,
        )


def test_padded_destination_matches_compact_scatter():
    block_size = 128
    tp_size = 4
    # sp_row_counts = [3, 2, 3, 1] -> max_rows=3, total=9 valid rows
    sp_row_counts = [3, 2, 3, 1]
    max_rows = max(sp_row_counts)
    total_rows = sum(sp_row_counts)

    # gather_compact_indices: rank-major padded buffer physical rows for each
    # global compressed row k (in global order).
    gather_compact_indices = [rank * max_rows + i for rank, count in enumerate(sp_row_counts) for i in range(count)]
    # 9 valid global compressed rows, each maps to a distinct cache slot.
    full_slot_mapping = np.array([[k, k % block_size] for k in range(total_rows)], dtype=np.int64)

    padded_dest = build_padded_destination(full_slot_mapping, gather_compact_indices, max_rows, tp_size, block_size)

    # Core invariant: valid rows point to the correct destination.
    for k, src in enumerate(gather_compact_indices):
        np.testing.assert_array_equal(padded_dest[src], full_slot_mapping[k])

    # Padding rows (not referenced by gather_compact_indices) are invalid.
    valid_rows = set(gather_compact_indices)
    for row in range(tp_size * max_rows):
        if row not in valid_rows:
            assert linear_slot(padded_dest[row], block_size) == -1, (
                f"padding row {row} should be invalid, got {padded_dest[row]}"
            )


def test_invalid_padding_rows_are_filtered():
    block_size = 128
    tp_size = 4
    max_rows = 2  # force padding rows (each rank has 1 valid row, padded to 2)
    gather_compact_indices = [rank * max_rows for rank in range(tp_size)]
    full_slot_mapping = np.array([[k, 0] for k in range(4)], dtype=np.int64)

    padded = build_padded_destination(full_slot_mapping, gather_compact_indices, max_rows, tp_size, block_size)
    # rows 1,3,5,7 are padding -> invalid [-1, block_size-1] -> linear -1
    for row in [1, 3, 5, 7]:
        assert linear_slot(padded[row], block_size) == -1
