# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.dcp_prefill import DCPReadLayout


def test_ragged_rank_major_pages():
    table = torch.tensor([[3, 1, 4], [2, -1, -1], [-1, -1, -1]])
    lengths = torch.tensor([513, 128, 0])
    plan = DCPReadLayout.build(table, lengths, 513, 128, 2)
    assert plan.local_page_ids.tolist() == [3, 1, 4, 2]

    class Pair:
        def all_gather(self, local, dim):
            return torch.cat((local, local + 100), dim=dim)

    gathered = plan.gather(torch.arange(5), Pair())
    assert gathered[plan.block_table[0, :5].long()].tolist() == [3, 103, 1, 101, 4]
    assert gathered[plan.block_table[1, :1].long()].tolist() == [2]
    assert plan.block_table[2].count_nonzero() == 0


def test_empty_read_and_short_table():
    table = torch.zeros((1, 1), dtype=torch.int)
    plan = DCPReadLayout.build(table, torch.tensor([0]), 0, 128, 2)
    assert plan.gather(torch.arange(1), None).numel() == 0
    with pytest.raises(ValueError, match="block table"):
        DCPReadLayout.build(table, torch.tensor([257]), 257, 128, 2)
