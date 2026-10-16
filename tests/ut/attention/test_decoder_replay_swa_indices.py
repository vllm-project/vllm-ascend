# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import build_replay_swa_visible_lens


@pytest.mark.parametrize(
    "query_start_loc, seq_lens, replay_start, expected_lengths",
    [
        pytest.param([0, 1, 5], [11, 101], [7, 97], [4, 1, 2, 3, 4], id="trimmed-prefill-and-decode"),
        pytest.param([0, 3, 4], [3, 11], [0, 0], [1, 2, 3, 4], id="without-replay"),
        pytest.param([0], [], [], [], id="empty-batch"),
    ],
)
def test_replay_swa_visible_lengths_bound_history(query_start_loc, seq_lens, replay_start, expected_lengths):
    # The attention operator consumes per-token lengths in band mode, keeping
    # decode history while excluding unwritten keys before the replay floor.
    lengths = build_replay_swa_visible_lens(
        window_size=4,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        replay_start=torch.tensor(replay_start, dtype=torch.int32),
        num_tokens=query_start_loc[-1],
    )
    torch.testing.assert_close(
        lengths,
        torch.tensor(expected_lengths, dtype=torch.int32).reshape(-1, 1),
    )
