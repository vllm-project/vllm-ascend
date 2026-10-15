# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.states import AscendRequestState


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("supports_dense_history", [False, True])
def test_request_state_preserves_dense_history_option(dense, supports_dense_history):
    """Enable device history for n-gram without requiring it on older releases."""
    counts = np.zeros(2, dtype=np.int32)

    # Use strict signatures so passing the new option to the old API fails.
    # This also covers release, RC, development and backported builds without
    # assuming their API from their version strings.
    def legacy_parent_init(
        self, max_num_reqs, max_model_len, max_num_batched_tokens, num_speculative_steps, vocab_size, device
    ):
        self.num_computed_tokens_np = counts

    dense_options = []

    def current_parent_init(
        self,
        max_num_reqs,
        max_model_len,
        max_num_batched_tokens,
        num_speculative_steps,
        vocab_size,
        device,
        use_dense_all_token_ids=False,
    ):
        self.num_computed_tokens_np = counts
        dense_options.append(use_dense_all_token_ids)

    parent_init = current_parent_init if supports_dense_history else legacy_parent_init
    with patch.object(RequestState, "__init__", parent_init):
        if not supports_dense_history and dense:
            with pytest.raises(ValueError, match="MRV2 n-gram requires vLLM"):
                AscendRequestState(2, 32, 8, 3, 16, torch.device("cpu"), use_dense_all_token_ids=True)
            return
        state = AscendRequestState(2, 32, 8, 3, 16, torch.device("cpu"), use_dense_all_token_ids=dense)
    if supports_dense_history:
        assert dense_options == [dense]
    state.num_computed_tokens_cpu[0] = 7
    assert counts[0] == 7
