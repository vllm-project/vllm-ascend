# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    _update_compressor_seqused,
    build_vision_bidirectional_swa_indices,
)


def _logical_slots(
    indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> list[int]:
    slot_to_pos = {
        block_id * block_size + block_offset: block_num * block_size + block_offset
        for block_num, block_id in enumerate(block_table.tolist())
        for block_offset in range(block_size)
    }
    return [slot_to_pos[slot] for slot in indices.tolist() if slot >= 0]


def test_compressor_seqused_masks_graph_padding_and_clears_stale_rows():
    buffer = torch.empty(6, dtype=torch.int32)
    graph_cu_seqlens = torch.arange(7, dtype=torch.int32)

    first = _update_compressor_seqused(buffer, graph_cu_seqlens, num_reqs=6, num_actual_reqs=4)
    assert first.tolist() == [1, 1, 1, 1, 0, 0]

    replay = _update_compressor_seqused(buffer, graph_cu_seqlens, num_reqs=6, num_actual_reqs=1)
    assert replay.tolist() == [1, 0, 0, 0, 0, 0]

    idle = _update_compressor_seqused(buffer, graph_cu_seqlens, num_reqs=6, num_actual_reqs=0)
    assert idle.tolist() == [0, 0, 0, 0, 0, 0]


def test_compressor_seqused_preserves_active_ragged_lengths():
    buffer = torch.empty(4, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5, 6, 7], dtype=torch.int32)

    result = _update_compressor_seqused(buffer, cu_seqlens, num_reqs=4, num_actual_reqs=2)

    assert result.tolist() == [2, 3, 0, 0]


def test_image_queries_see_complete_block_and_text_stays_causal():
    block_size = 4
    block_table = torch.tensor([[7, 2, 9]], dtype=torch.int32)

    indices, _ = build_vision_bidirectional_swa_indices(
        block_table=block_table,
        window_size=3,
        max_image_tokens=5,
        block_size=block_size,
        query_start_loc=torch.tensor([0, 12], dtype=torch.int32),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        mm_prefix_ranges={0: [(4, 8)]},
        num_tokens=12,
    )

    assert _logical_slots(indices[3, 0], block_table[0], block_size) == [1, 2, 3]
    assert _logical_slots(indices[4, 0], block_table[0], block_size) == list(range(2, 9))
    assert _logical_slots(indices[6, 0], block_table[0], block_size) == list(range(4, 9))
    assert _logical_slots(indices[8, 0], block_table[0], block_size) == list(range(4, 9))
    assert _logical_slots(indices[9, 0], block_table[0], block_size) == [7, 8, 9]


def test_multiple_images_and_batches_use_their_own_physical_blocks():
    block_size = 4
    block_table = torch.tensor([[3, 8], [11, 5]], dtype=torch.int32)

    indices, _ = build_vision_bidirectional_swa_indices(
        block_table=block_table,
        window_size=2,
        max_image_tokens=4,
        block_size=block_size,
        query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        mm_prefix_ranges={0: [(1, 2), (5, 7)], 1: [(3, 6)]},
        num_tokens=16,
    )

    assert _logical_slots(indices[5, 0], block_table[0], block_size) == [4, 5, 6, 7]
    assert _logical_slots(indices[11, 0], block_table[1], block_size) == [2, 3, 4, 5, 6]


def test_image_span_must_fit_current_prefill_chunk():
    with pytest.raises(ValueError, match="single prefill chunk"):
        build_vision_bidirectional_swa_indices(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            window_size=2,
            max_image_tokens=4,
            block_size=4,
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            mm_prefix_ranges={0: [(2, 5)]},
            num_tokens=4,
        )
