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
from vllm.pooling_params import PoolingParams
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

from vllm_ascend.platform import NPUPlatform


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

    state = ThinkingBudgetState(SimpleNamespace(max_num_reqs=2, device=device), None)
    assert not state.enabled
    assert not state.add_request(0, SamplingParams(thinking_token_budget=budget))
    state.apply_staged_writes()
    logits = torch.tensor([[0.0, 1.0, 1.0]])
    expected = logits.clone()
    state.apply(logits, SimpleNamespace(idx_mapping_np=np.array([0], dtype=np.int32)))
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("value", [None, -1, 0, 2**31, 2**63, 2**64 - 1])
def test_platform_accepts_transportable_budget(value):
    NPUPlatform.validate_request({}, SamplingParams(thinking_token_budget=value))


@pytest.mark.parametrize("value", [2**64, 2**80, 10**5000], ids=["uint64-overflow", "uint80", "huge-integer"])
def test_platform_rejects_budget_transport_overflow(value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget must be at most") as error:
        NPUPlatform.validate_request({}, SamplingParams(thinking_token_budget=value))
    assert error.value.parameter == "thinking_token_budget"
    assert error.value.value is None
    assert "thinking_token_budget" in str(error.value)


def test_platform_budget_validation_preserves_pooling_requests():
    NPUPlatform.validate_request({}, PoolingParams())
