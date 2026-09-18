# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_ascend.spec_decode.mtp import compact_mtp_topk_indices


@pytest.mark.parametrize(
    "num_input_tokens,sample_ids,world_size",
    [(8, [3, 7], 2), (7, [0, 2, 6], 2), (12, [1, 5, 10], 4), (4, [0, 1, 2, 3], 2)],
)
def test_dsa_cp_compaction_moves_rows_to_next_step_owner(num_input_tokens, sample_ids, world_size):
    local_tokens = (num_input_tokens + world_size - 1) // world_size
    all_rows = torch.arange(local_tokens * world_size * 4, dtype=torch.int32).reshape(-1, 4)
    all_rows[:, -1] = -1
    sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
    expected_rows = all_rows[sample_ids]

    for rank in range(world_size):
        start = rank * local_tokens
        model = nn.Module()
        model.topk_indices_buffer = torch.full_like(all_rows, -99)
        model.topk_indices_buffer[:local_tokens].copy_(all_rows[start : start + local_tokens])
        # The predictor and MLA wrappers can expose the same underlying buffer.
        model.add_module("attention", nn.Module())
        model.attention.topk_indices_buffer = model.topk_indices_buffer
        original_buffer = model.topk_indices_buffer
        owns_row = (sample_ids >= start) & (sample_ids < start + local_tokens)

        def reduce_rows(rows, owns_row=owns_row):
            torch.testing.assert_close(rows, expected_rows.masked_fill(~owns_row[:, None], 0))
            return expected_rows.clone()

        group = SimpleNamespace(world_size=world_size, rank_in_group=rank, all_reduce=Mock(side_effect=reduce_rows))
        compact_mtp_topk_indices(model, sample_ids, num_input_tokens, group)

        group.all_reduce.assert_called_once()
        assert model.topk_indices_buffer is original_buffer
        count = max(0, min(local_tokens, len(sample_ids) - start))
        torch.testing.assert_close(original_buffer[:count], expected_rows[start : start + count])


def test_compaction_without_dsa_cp_uses_predictor_hook():
    model = nn.Module()
    model.compact_topk_indices = Mock()
    indices = torch.tensor([1, 5], dtype=torch.int32)
    compact_mtp_topk_indices(model, indices, 6)
    model.compact_topk_indices.assert_called_once_with(indices)


def test_dsa_cp_compaction_empty_batch_skips_collective():
    group = SimpleNamespace(all_reduce=Mock())
    compact_mtp_topk_indices(nn.Module(), torch.empty(0, dtype=torch.int32), 0, group)
    group.all_reduce.assert_not_called()
