#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is part of the vllm-ascend project.
"""Unit tests for AscendSpecDecodeBaseProposer._split_pcp_input_hybrid."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _build_inputs(req_scheduled_tokens: dict[str, int], hidden_size: int):
    """Build input_ids and target_hidden_states where row i is filled with i.

    Using the row index as the value makes it easy to assert which original
    tokens survive the per-rank slice and which slots are padding.
    """
    total_tokens = sum(req_scheduled_tokens.values())
    input_ids = torch.arange(total_tokens, dtype=torch.int32)
    target_hidden_states = (
        torch.arange(total_tokens, dtype=torch.float32)
        .unsqueeze(-1)
        .repeat(1, hidden_size)
    )
    return input_ids, target_hidden_states


# yapf: disable
@pytest.mark.parametrize(
    "pcp_size, pcp_rank, req_scheduled_tokens, hidden_size,"
    " expected_num_tokens, expected_input_ids, expected_hidden_first_col,"
    " expected_seq_lens, expected_cu_num_tokens, expected_max_query_len",
    [
        # Case 1: single req, perfectly aligned to 2*pcp_size.
        # ori=8, pcp=2 -> padded=8, pcp_tokens=4
        # rank 0: [0,1,2,3], rank 1: [4,5,6,7]
        (
            2, 0, {"0": 8}, 4,
            4, [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0],
            [4], [0, 4], 4,
        ),
        (
            2, 1, {"0": 8}, 4,
            4, [4, 5, 6, 7], [4.0, 5.0, 6.0, 7.0],
            [4], [0, 4], 4,
        ),
        # Case 2: single req, needs padding.
        # ori=7, pcp=2 -> padded=8, pcp_tokens=4, num_pads=1
        # rank 0: [0,1,2,3] (all valid)
        # rank 1: [4,5,6,PAD] -> input_ids=[4,5,6,0], hidden_first_col=[4,5,6,0]
        (
            2, 0, {"0": 7}, 4,
            4, [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0],
            [4], [0, 4], 4,
        ),
        (
            2, 1, {"0": 7}, 4,
            4, [4, 5, 6, 0], [4.0, 5.0, 6.0, 0.0],
            [4], [0, 4], 4,
        ),
        # Case 3: multiple reqs with different padding needs.
        # req 0: ori=4 -> padded=4, pcp_tokens=2
        # req 1: ori=6 -> padded=8, pcp_tokens=4, num_pads=2
        # rank 0: [0,1] + [4,5,6,7]
        # rank 1: [2,3] + [8,9,PAD,PAD]
        (
            2, 0, {"0": 4, "1": 6}, 2,
            6, [0, 1, 4, 5, 6, 7], [0.0, 1.0, 4.0, 5.0, 6.0, 7.0],
            [2, 4], [0, 2, 6], 4,
        ),
        (
            2, 1, {"0": 4, "1": 6}, 2,
            6, [2, 3, 8, 9, 0, 0], [2.0, 3.0, 8.0, 9.0, 0.0, 0.0],
            [2, 4], [0, 2, 6], 4,
        ),
        # Case 4: pcp_size=4
        # ori=9, pcp=4 -> padded=16, pcp_tokens=4, num_pads=7
        # rank 0: tokens [0,1,2,3]
        # rank 1: tokens [4,5,6,7]
        # rank 2: tokens [8,PAD,PAD,PAD]
        # rank 3: tokens [PAD,PAD,PAD,PAD]
        (
            4, 0, {"0": 9}, 2,
            4, [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0],
            [4], [0, 4], 4,
        ),
        (
            4, 1, {"0": 9}, 2,
            4, [4, 5, 6, 7], [4.0, 5.0, 6.0, 7.0],
            [4], [0, 4], 4,
        ),
        (
            4, 2, {"0": 9}, 2,
            4, [8, 0, 0, 0], [8.0, 0.0, 0.0, 0.0],
            [4], [0, 4], 4,
        ),
        (
            4, 3, {"0": 9}, 2,
            4, [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0],
            [4], [0, 4], 4,
        ),
        # Case 5: minimal request - single token.
        # ori=1, pcp=2 -> padded=4, pcp_tokens=2, num_pads=3
        # rank 0: [0, PAD]
        # rank 1: [PAD, PAD]
        (
            2, 0, {"0": 1}, 2,
            2, [0, 0], [0.0, 0.0],
            [2], [0, 2], 2,
        ),
        (
            2, 1, {"0": 1}, 2,
            2, [0, 0], [0.0, 0.0],
            [2], [0, 2], 2,
        ),
    ],
)
# yapf: enable
def test_split_pcp_input_hybrid(
    pcp_size,
    pcp_rank,
    req_scheduled_tokens,
    hidden_size,
    expected_num_tokens,
    expected_input_ids,
    expected_hidden_first_col,
    expected_seq_lens,
    expected_cu_num_tokens,
    expected_max_query_len,
):
    input_ids, target_hidden_states = _build_inputs(req_scheduled_tokens, hidden_size)

    # _split_pcp_input_hybrid only reads self.pcp_size and self.pcp_rank,
    # so a MagicMock with just those attributes is enough to drive the
    # unbound method without instantiating the full proposer.
    mock_self = MagicMock()
    mock_self.pcp_size = pcp_size
    mock_self.pcp_rank = pcp_rank

    (
        num_tokens,
        out_input_ids,
        out_hidden_states,
        max_query_len,
        seq_lens,
        cu_num_tokens,
    ) = AscendSpecDecodeBaseProposer._split_pcp_input_hybrid(
        mock_self, req_scheduled_tokens, input_ids, target_hidden_states
    )

    assert num_tokens == expected_num_tokens
    assert max_query_len == expected_max_query_len
    assert torch.equal(
        out_input_ids, torch.tensor(expected_input_ids, dtype=torch.int32)
    )
    assert torch.equal(seq_lens, torch.tensor(expected_seq_lens, dtype=torch.int32))
    assert torch.equal(
        cu_num_tokens, torch.tensor(expected_cu_num_tokens, dtype=torch.int64)
    )

    # hidden_states shape: [num_tokens, hidden_size]
    assert out_hidden_states.shape == (expected_num_tokens, hidden_size)
    assert torch.equal(
        out_hidden_states[:, 0],
        torch.tensor(expected_hidden_first_col, dtype=torch.float32),
    )


def test_split_pcp_input_hybrid_preserves_hidden_size():
    """Hidden states must come back with the same hidden dim as the input."""
    hidden_size = 7
    req_scheduled_tokens = {"0": 6}
    input_ids, target_hidden_states = _build_inputs(req_scheduled_tokens, hidden_size)

    mock_self = MagicMock()
    mock_self.pcp_size = 2
    mock_self.pcp_rank = 0

    _, _, out_hidden_states, _, _, _ = (
        AscendSpecDecodeBaseProposer._split_pcp_input_hybrid(
            mock_self, req_scheduled_tokens, input_ids, target_hidden_states
        )
    )

    assert out_hidden_states.shape[1] == hidden_size
