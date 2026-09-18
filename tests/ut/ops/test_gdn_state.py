# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn_state import canonicalize_folded_prefill_state


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("num_tokens", [2, 4, 8])
@pytest.mark.parametrize("page_padding", [0, 5])
def test_folded_prompt_commits_both_states_across_chunks(dim_first, num_tokens, page_padding):
    # Physical blocks are deliberately non-contiguous and include page padding.
    blocks = [11, 3, 14, 6, 12, 4, 9, 2, 13, 1]
    column = 1
    first, last = blocks[column], blocks[column + num_tokens - 1]
    history_len, dim = 3, 5
    state_len = history_len + num_tokens - 1
    shape = (dim, state_len) if dim_first else (state_len, dim)
    storage = torch.full((16, dim * state_len + page_padding), -100.0)
    conv = storage[:, : dim * state_len].view(16, *shape)
    recurrent = torch.full((16, 2, 3), -200.0)
    history_axis = 1 if dim_first else 0
    with patch("vllm_ascend.ops.gdn_state.is_conv_state_dim_first", return_value=dim_first):
        # Two consecutive folded chunks, followed by a non-spec read at slot 0.
        for step in range(2):
            expected_state = torch.full((2, 3), 40.0 + step)
            expected_history = torch.arange(history_len * dim, dtype=torch.float32).view(history_len, dim) + step * 100
            if dim_first:
                expected_history = expected_history.T
            recurrent[last].copy_(expected_state)
            conv[first].narrow(history_axis, num_tokens - 1, history_len).copy_(expected_history)
            before_conv, before_recurrent = conv.clone(), recurrent.clone()
            canonicalize_folded_prefill_state(
                conv, recurrent, torch.tensor(blocks[column : column + num_tokens]), num_tokens
            )
            torch.testing.assert_close(recurrent[first], expected_state)
            torch.testing.assert_close(conv[first].narrow(history_axis, 0, history_len), expected_history)
            # Other requests and uncommitted slots must not be changed.
            untouched = [i for i in range(16) if i != first]
            torch.testing.assert_close(conv[untouched], before_conv[untouched])
            torch.testing.assert_close(recurrent[untouched], before_recurrent[untouched])
            if page_padding:
                assert (storage[:, -page_padding:] == -100).all()
