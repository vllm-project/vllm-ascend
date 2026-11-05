# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.v1.sample.thinking_budget_state import maybe_create_thinking_budget_state_holder
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState


@pytest.mark.parametrize("value", [None, -1, 0, 1, 64, 2**31 - 1, 2**31, 2**40])
def test_budget_validation_agrees_between_offline_and_openai(value):
    expected = None if value == -1 else value
    offline = SamplingParams(thinking_token_budget=value)
    chat = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": "test"}], thinking_token_budget=value
    )
    completion = CompletionRequest(model="test", prompt="test", thinking_token_budget=value)
    assert offline.thinking_token_budget == chat.thinking_token_budget == completion.thinking_token_budget == expected


@pytest.mark.parametrize("value", [-2, True, False, 1.0, 1.5, "3", [], {}])
@pytest.mark.parametrize("kind", ["offline", "chat", "completion"])
def test_invalid_budget_fails_before_engine_execution(value, kind):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        if kind == "offline":
            SamplingParams(thinking_token_budget=value)
        elif kind == "chat":
            ChatCompletionRequest(
                model="test", messages=[{"role": "user", "content": "test"}], thinking_token_budget=value
            )
        else:
            CompletionRequest(model="test", prompt="test", thinking_token_budget=value)


@pytest.mark.parametrize("budget", [None, -1, 0, 16])
def test_budget_state_is_disabled_without_reasoning_config(budget):
    device = torch.device("cpu")
    assert maybe_create_thinking_budget_state_holder(None, 2, 3, device, False) is None

    state = ThinkingBudgetState(SimpleNamespace(max_num_reqs=2, device=device), None)
    assert not state.enabled
    assert not state.add_request(0, SamplingParams(thinking_token_budget=budget))
    state.apply_staged_writes()
    logits = torch.tensor([[0.0, 1.0, 1.0]])
    expected = logits.clone()
    state.apply(logits, SimpleNamespace(idx_mapping_np=np.array([0], dtype=np.int32)))
    assert torch.equal(logits, expected)
