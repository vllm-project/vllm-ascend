# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.mla_draft_window import MLADraftWindow


@pytest.mark.parametrize("long_len", [2050, 2051, 2052, 2178, 2179, 2180, 65_537])
def test_mla_draft_window_keeps_the_same_tail_for_all_draft_queries(long_len):
    window_size, block_size, query_width = 2051, 128, 3
    num_pages = (long_len + block_size - 1) // block_size
    table = torch.stack((torch.arange(1, num_pages + 1), torch.arange(1, num_pages + 1) + num_pages))
    lengths = [long_len, 47]
    window = MLADraftWindow(window_size, block_size, query_width)

    window_table, mask, window_lengths = window.prepare(table, lengths, num_reqs=3)

    assert mask.shape == (3, query_width, 17 * block_size + block_size)
    assert mask.is_contiguous()
    assert window_lengths[-1] == 0
    assert window_table[-1].count_nonzero() == 0
    for index, length in enumerate(lengths):
        offsets = torch.arange(block_size)
        original_slots = (table[index, :, None] * block_size + offsets).flatten()
        window_slots = (window_table[index, :, None] * block_size + offsets).flatten()
        valid_length = window_lengths[index]
        expected = original_slots[max(0, length - window_size) : length]
        for query in range(query_width):
            visible = window_slots[:valid_length][~mask[index, query, :valid_length]]
            torch.testing.assert_close(visible, expected)


def test_mla_draft_window_rebuilds_replay_inputs_and_keeps_their_addresses():
    window = MLADraftWindow(window_size=7, block_size=4, query_width=3)
    table = torch.tensor([[5, 7, 11, 13, 17]], dtype=torch.int32)
    capture_table, capture_mask, _ = window.prepare(table, [3], num_reqs=1, capture=True)

    replay_table, replay_mask, lengths = window.prepare(table, [16], num_reqs=1, replay=True)
    assert capture_table.data_ptr() != replay_table.data_ptr()
    assert capture_mask.data_ptr() != replay_mask.data_ptr()
    torch.testing.assert_close(replay_table, table[:, 2:5])
    assert lengths == [8]
    assert replay_mask[0, :, 0].all()
    assert not replay_mask[0, :, 1:].any()

    updated_table, updated_mask, lengths = window.prepare(table, [4], num_reqs=1, replay=True)
    assert updated_table.data_ptr() == replay_table.data_ptr()
    assert updated_mask.data_ptr() == replay_mask.data_ptr()
    torch.testing.assert_close(updated_table, table[:, :3])
    assert lengths == [4]
    assert not updated_mask.any()
