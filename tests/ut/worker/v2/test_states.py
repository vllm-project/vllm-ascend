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
    """Enable device history for n-gram without requiring it on older releases."""
    counts = np.zeros(2, dtype=np.int32)

    def parent_init(self, *args, **kwargs):
        self.num_computed_tokens_np = counts

    with patch.object(RequestState, "__init__", autospec=True, side_effect=parent_init) as parent:
        state = AscendRequestState(2, 32, 8, 3, 16, torch.device("cpu"), use_dense_all_token_ids=dense)
    assert parent.call_args.kwargs == ({"use_dense_all_token_ids": True} if dense else {})
    state.num_computed_tokens_cpu[0] = 7
    assert counts[0] == 7
