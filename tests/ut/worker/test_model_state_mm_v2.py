# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


@pytest.mark.parametrize("state_cls", [AscendModelState, AscendMambaHybridModelState])
@pytest.mark.parametrize("num_tokens", [0, 7])
def test_gather_without_local_encoder(state_cls, num_tokens):
    state = state_cls.__new__(state_cls)
    state.supports_mm_inputs = False
    batch = SimpleNamespace(num_tokens=num_tokens, num_tokens_after_padding=16)

    with patch.object(DefaultModelState, "gather_mm_embeddings") as gather:
        embeddings, mask = state.gather_mm_embeddings(batch, draft_lookahead=1)

    gather.assert_not_called()
    assert not hasattr(state, "encoder_runner")
    assert embeddings == []
    assert mask.shape == (num_tokens,)
    assert mask.dtype == torch.bool
    assert mask.device.type == "cpu"
    assert not mask.any()


@pytest.mark.parametrize("state_cls", [AscendModelState, AscendMambaHybridModelState])
@pytest.mark.parametrize("draft_lookahead", [0, 1])
def test_gather_with_local_encoder_delegates(state_cls, draft_lookahead):
    state = state_cls.__new__(state_cls)
    state.supports_mm_inputs = True
    batch = SimpleNamespace(num_tokens=2)
    expected = ([torch.ones(1, 4)], torch.tensor([False, True]))

    with patch.object(DefaultModelState, "gather_mm_embeddings", return_value=expected) as gather:
        result = state.gather_mm_embeddings(batch, draft_lookahead=draft_lookahead)

    gather.assert_called_once_with(batch, draft_lookahead)
    assert result is expected
