# SPDX-License-Identifier: Apache-2.0
"""CPU host-state contracts; these tests do not execute Triton/NPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample import thinking_budget


@pytest.fixture
def budget_state(monkeypatch):
    class CPUBuffer:
        def __init__(self, size, dtype):
            self.gpu = torch.empty(size, dtype=dtype)
            self.np = self.gpu.numpy()
            self.copy_to_uva = Mock()

    monkeypatch.setattr(thinking_budget, "UvaBackedTensor", CPUBuffer)
    monkeypatch.setattr(thinking_budget, "async_tensor_h2d", lambda values, **kwargs: torch.tensor(values, **kwargs))
    requests = SimpleNamespace(
        max_num_reqs=2,
        device=torch.device("cpu"),
        all_token_ids=SimpleNamespace(gpu=torch.zeros((2, 16), dtype=torch.int32)),
        total_len=SimpleNamespace(gpu=torch.tensor([8, 12], dtype=torch.int32)),
    )
    config = SimpleNamespace(
        reasoning_start_token_ids=[11, 12],
        natural_reasoning_end_token_ids=[13, 14],
        reasoning_end_token_ids=[13, 14, 15],
    )
    return thinking_budget.ThinkingBudgetState(requests, config)


@pytest.mark.parametrize("budget,expected", [(None, -1), (0, 0), (16, 16), (2**40, 2**31 - 1)])
def test_host_budget_request_reuse_and_staging(budget_state, budget, expected):
    state = budget_state
    state.cached_last_start.fill_(7)
    state.cached_last_end.fill_(9)
    state.cached_scan_pos.fill_(12)
    state.add_request(0, SimpleNamespace(thinking_token_budget=budget))
    state.apply_staged_writes()
    assert state.thinking_token_budget.np.tolist() == [expected, -1]
    assert state.use_thinking_budget.tolist() == [budget is not None, False]
    if budget is not None:
        assert state.cached_last_start.tolist() == [-1, 7]
        assert state.cached_last_end.tolist() == [-1, 9]
        assert state.cached_scan_pos.tolist() == [0, 12]
    copies = state.thinking_token_budget.copy_to_uva.call_count
    state.apply_staged_writes()
    assert state.thinking_token_budget.copy_to_uva.call_count == copies
    state.add_request(0, SimpleNamespace(thinking_token_budget=None))
    state.apply_staged_writes()
    assert state.thinking_token_budget.np[0] == -1
    assert not state.use_thinking_budget[0]


def test_host_budget_apply_preserves_upstream_kernel_inputs(budget_state, monkeypatch):
    state = budget_state
    apply = Mock()
    monkeypatch.setattr(thinking_budget, "apply_thinking_budget", apply)
    args = (
        torch.zeros((2, 32)),
        torch.tensor([0, 0]),
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([42, 43]),
        torch.tensor([0, 1]),
    )
    state.apply(*args)
    apply.assert_not_called()
    state.add_request(0, SimpleNamespace(thinking_token_budget=5))
    state.apply_staged_writes()
    state.apply(*args)
    apply.assert_called_once()
    passed = apply.call_args.args
    assert passed[0] is args[0]
    assert passed[3] is state.thinking_token_budget.gpu
    assert passed[4] is state.req_states.all_token_ids.gpu
    assert passed[6] is args[4]
    assert passed[7] is args[5]


def test_host_budget_disabled_without_markers():
    state = thinking_budget.ThinkingBudgetState(SimpleNamespace(max_num_reqs=2, device=torch.device("cpu")), None)
    assert not state.enabled
    state.add_request(0, SimpleNamespace(thinking_token_budget=0))
    state.apply_staged_writes()
