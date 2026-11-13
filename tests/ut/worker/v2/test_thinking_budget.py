# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""V2 host state contracts; device budget arithmetic is covered by NPU E2E."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.worker.gpu import states as request_states
from vllm.v1.worker.gpu.sample import thinking_budget
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.worker.v2.states import AscendRequestState

START = 90
END = 91
MAX_REQUESTS = 4


class HostBuffer:
    """Replace only staged device storage, retaining the real request/state code."""

    def __init__(self, size, dtype, **kwargs):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu.clone()

    def copy_to_uva(self):
        self.gpu.copy_(self.cpu)
        return self.gpu

    def stage_write_elem(self, row, value):
        self.cpu[row] = value

    def stage_write(self, row, offset, values):
        self.cpu[row, offset : offset + len(values)] = torch.tensor(values)

    def apply_write(self):
        self.copy_to_uva()


@pytest.fixture
def state_pair(monkeypatch):
    monkeypatch.setattr(request_states, "UvaBackedTensor", HostBuffer)
    monkeypatch.setattr(request_states, "StagedWriteTensor", HostBuffer)
    monkeypatch.setattr(thinking_budget, "UvaBackedTensor", HostBuffer)
    monkeypatch.setattr(thinking_budget, "async_tensor_h2d", lambda data, **kwargs: torch.tensor(data, **kwargs))
    requests = AscendRequestState(MAX_REQUESTS, 64, 16, 3, 128, torch.device("cpu"))
    config = SimpleNamespace(
        reasoning_start_token_ids=[START], reasoning_end_token_ids=[END], natural_reasoning_end_token_ids=[END]
    )
    return requests, thinking_budget.ThinkingBudgetState(requests, config)


def add_request(pair, name, budget, history=(1, START)):
    requests, state = pair
    requests.add_request(name, 2, list(history), len(history), 32)
    index = requests.req_id_to_index[name]
    state.add_request(index, SamplingParams(thinking_token_budget=budget))
    requests.apply_staged_writes()
    state.apply_staged_writes()
    return index


@pytest.mark.parametrize("budget,stored", [(None, -1), (-1, -1), (0, 0), (1, 1), (64, 64), (2**40, 2**31 - 1)])
def test_budget_registration_and_device_staging(state_pair, budget, stored):
    _, state = state_pair
    index = add_request(state_pair, "req", budget)
    assert state.thinking_token_budget.gpu[index] == stored
    assert bool(state.use_thinking_budget[index]) == (stored >= 0)


def test_removed_request_and_reused_slot_do_not_change_other_budgets(state_pair):
    requests, state = state_pair
    first = add_request(state_pair, "zero", 0)
    second = add_request(state_pair, "long", 16)
    third = add_request(state_pair, "short", 3)
    state.cached_last_start[second] = 1
    state.cached_scan_pos[second] = 12
    assert requests.remove_request("zero") == first
    assert add_request(state_pair, "unlimited", None) == first
    assert state.thinking_token_budget.gpu[[third, first, second]].tolist() == [3, -1, 16]
    assert state.cached_scan_pos[second] == 12
    for name in list(requests.req_id_to_index):
        requests.remove_request(name)
    assert requests.num_reqs == 0
    index = add_request(state_pair, "new", 0)
    assert state.cached_last_start[index] == -1
    assert state.cached_last_end[index] == -1
    assert state.cached_scan_pos[index] == 0


@pytest.mark.parametrize("history", [(1, START, 10, 11), (1, START, 10, END), (1, START, 10, 92)])
def test_preemption_reconstructs_history_and_resets_marker_cache(state_pair, history):
    requests, state = state_pair
    slot = add_request(state_pair, "resumed", 2, history)
    state.cached_last_start[slot] = 40
    state.cached_last_end[slot] = 45
    state.cached_scan_pos[slot] = 50
    requests.remove_request("resumed")
    # Another request occupies the old slot before the preempted one resumes.
    assert add_request(state_pair, "replacement", 16) == slot
    resumed = add_request(state_pair, "resumed", 2, history)
    assert resumed != slot
    assert requests.prompt_len.gpu[resumed] == 2
    assert requests.prefill_len.gpu[resumed] == len(history)
    assert requests.total_len.gpu[resumed] == len(history)
    assert requests.all_token_ids.gpu[resumed, : len(history)].tolist() == list(history)
    assert state.cached_last_start[resumed] == -1
    assert state.cached_last_end[resumed] == -1
    assert state.cached_scan_pos[resumed] == 0
    assert state.thinking_token_budget.gpu[[slot, resumed]].tolist() == [16, 2]


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_batch_reorder_passes_request_slots_to_budget_kernel(state_pair, monkeypatch, order):
    requests, state = state_pair
    slots = [add_request(state_pair, str(i), budget) for i, budget in enumerate([0, 3, None])]
    mapping = torch.tensor([slots[i] for i in order], dtype=torch.int32)
    expanded = mapping.repeat_interleave(torch.tensor([1, 3, 2]))
    local_pos = torch.tensor([0, 0, 1, 2, 0, 1], dtype=torch.int32)
    input_ids = torch.tensor([START, 10, 11, 12, 20, 21], dtype=torch.int32)
    launch = Mock()
    monkeypatch.setattr(thinking_budget, "apply_thinking_budget", launch)
    state.apply(
        torch.zeros(6, 128),
        SimpleNamespace(
            idx_mapping=mapping,
            idx_mapping_np=mapping.numpy(),
            expanded_idx_mapping=expanded,
            expanded_local_pos=local_pos,
            input_ids=input_ids,
        ),
    )
    args = launch.call_args.args
    assert args[1] is mapping
    assert args[2] is expanded
    assert args[4] is requests.all_token_ids.gpu
    assert args[6] is input_ids
    assert args[7] is local_pos


def test_empty_or_unlimited_batch_bypasses_kernel(state_pair, monkeypatch):
    _, state = state_pair
    slot = add_request(state_pair, "plain", None)
    launch = Mock()
    monkeypatch.setattr(thinking_budget, "apply_thinking_budget", launch)
    for indices in ([], [slot]):
        state.apply(torch.zeros(len(indices), 128), SimpleNamespace(idx_mapping_np=np.array(indices, dtype=np.int32)))
    launch.assert_not_called()


@pytest.mark.parametrize("skip_top_k_top_p", [False, True])
def test_sampler_applies_budget_after_processors_for_normal_and_verification_rows(skip_top_k_top_p):
    """Both ordinary sampling and rejection verification use this real method."""
    sampler = Sampler.__new__(Sampler)
    sampler.needs_logits_processing = np.array([True, False])
    events = []

    def mask_end(logits, ctx):
        events.append("processor")
        logits[:, END] = -torch.inf
        return logits

    def force_end(logits, ctx):
        events.append("budget")
        assert torch.isneginf(logits[:, END]).all()
        logits[ctx.expanded_idx_mapping == 0, END] = 1e9

    sampler.logits_processors = [SimpleNamespace(apply=mask_end)]
    sampler.thinking_budget_state = SimpleNamespace(apply=force_end)
    sampler.sampling_states = SimpleNamespace(
        apply_temperature=lambda *args: events.append("temperature"),
        apply_min_p=lambda *args: events.append("min_p"),
        apply_top_k_top_p=lambda logits, *args: logits,
    )
    expanded = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
    result = sampler.apply_sampling_params(
        torch.zeros(4, 128),
        expanded,
        torch.tensor([1, 0]),
        np.array([1, 0]),
        torch.zeros(4),
        torch.tensor([10, 10, 11, 12]),
        torch.tensor([0, 0, 1, 2]),
        np.array([3, 5]),
        skip_top_k_top_p=skip_top_k_top_p,
    )
    assert events == ["processor", "budget", "temperature", "min_p"]
    assert torch.isneginf(result[0, END])
    assert result[1:, END].tolist() == [1e9] * 3
