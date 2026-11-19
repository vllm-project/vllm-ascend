# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.states import AscendRequestState


@pytest.mark.parametrize("dense", [False, True])
def test_request_state_preserves_dense_history_option(dense):
    """Forward the dense history option to the upstream request state."""
    counts = np.zeros(2, dtype=np.int32)
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

    with patch.object(RequestState, "__init__", current_parent_init):
        state = AscendRequestState(2, 32, 8, 3, 16, torch.device("cpu"), use_dense_all_token_ids=dense)
    assert dense_options == [dense]
    state.num_computed_tokens_cpu[0] = 7
    assert counts[0] == 7
