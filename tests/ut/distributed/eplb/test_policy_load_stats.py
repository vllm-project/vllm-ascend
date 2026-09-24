# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.distributed.eplb import state as state_module
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.distributed.eplb.state import AscendEplbState


def _custom_policy():
    return SimpleNamespace(prepare_local_load_stats=lambda samples: PreparedLoadStats(samples))


def _model_state(**kwargs):
    defaults = dict(
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_step_keeps_skipped_samples_on_shared_time_axis(monkeypatch):
    upstream_step = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = _custom_policy()
    state.model_states = {"model": _model_state()}
    state.expert_load_window_size = 2
    state.expert_load_window_step = 0
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((2,), -1)
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._is_load_sampling_step = True
    state._should_collect_local_load = False

    state.step()
    state.expert_load_window_step = 1
    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.step()

    torch.testing.assert_close(
        state._local_load_collection_mask,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(state._physical_load_sample_slots, torch.tensor([-1, 1]))
    assert state._num_recorded_load_steps == 2
    assert upstream_step.call_count == 2


def test_mapping_change_discards_old_samples():
    state = AscendEplbState.__new__(AscendEplbState)
    model_state = _model_state(
        _load_mapping_generation=2,
        _observed_load_mapping_generation=1,
    )
    state.model_states = {"model": model_state}
    state._local_load_collection_mask = torch.ones(3, dtype=torch.int32)
    state._physical_load_sample_slots = torch.arange(3)
    state._num_recorded_load_steps = 3
    state._load_stats_window_start_index = 1
    state._load_stats_window_write_index = 2

    state._discard_samples_from_old_mapping()

    assert not state._local_load_collection_mask.any()
    assert bool((state._physical_load_sample_slots == -1).all())
    assert state._num_recorded_load_steps == 0
    assert model_state._observed_load_mapping_generation == 2


def test_collect_global_load_stats_maps_physical_to_logical(monkeypatch):
    cpu_group = object()
    device_group = object()
    monkeypatch.setattr(
        state_module,
        "get_eplb_group",
        lambda: SimpleNamespace(cpu_group=cpu_group, device_group=device_group),
    )
    all_reduce = MagicMock()
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)
    model_state = _model_state(
        model=SimpleNamespace(num_logical_experts=2),
        physical_to_logical_map=torch.tensor([[1, 0, 1]]),
        expert_load_window=torch.tensor([[[2, 3, 5]], [[7, 11, 13]], [[17, 19, 23]]]),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = _custom_policy()
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 3
    state._local_load_collection_mask = torch.tensor([1, 0, 1])
    state._physical_load_sample_slots = torch.tensor([0, -1, 2])
    state._num_recorded_load_steps = 3
    state._load_stats_window_start_index = 0

    result = state.collect_global_load_stats()

    assert result is not None
    torch.testing.assert_close(
        result["model"].values,
        torch.tensor([[[3, 7]], [[19, 40]]]),
    )
    assert [call.kwargs["group"] for call in all_reduce.call_args_list] == [
        cpu_group,
        device_group,
    ]


def test_collect_global_load_stats_skips_empty_window(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = _custom_policy()
    state.model_states = {"model": _model_state()}
    state._num_recorded_load_steps = 0
    state._discard_samples_from_old_mapping = MagicMock()
    get_group = MagicMock()
    monkeypatch.setattr(state_module, "get_eplb_group", get_group)

    assert state.collect_global_load_stats() is None
    get_group.assert_not_called()


def test_publish_async_load_stats_is_atomic(monkeypatch):
    device_group = MagicMock()
    device_group.size.return_value = 4
    monkeypatch.setattr(
        state_module,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=device_group),
    )
    monkeypatch.setattr(state_module, "get_node_count", lambda: 2)
    model_state = _model_state(
        model=SimpleNamespace(
            num_physical_experts=8,
            num_expert_groups=1,
        ),
        rebalanced=False,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.rearrange_event = MagicMock()
    stats = PreparedLoadStats(torch.ones(2, 1, 3), np.array([2, 1]))

    state.publish_async_load_stats({"model": stats})

    assert model_state._policy_load_stats is stats
    assert model_state.eplb_stats.num_nodes == 2
    assert model_state.eplb_stats.num_gpus == 4
    assert model_state.rebalanced
    state.rearrange_event.record.assert_called_once_with()
