# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats


def _custom_policy():
    return SimpleNamespace(prepare_local_load_stats=lambda samples: PreparedLoadStats(samples))


def _model_state(**kwargs):
    defaults = dict(
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _step_state(monkeypatch, **overrides):
    """A state with a two-slot window whose upstream step is mocked."""
    upstream_step = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state.policy = _custom_policy()
    state.model_states = {"model": _model_state()}
    state.expert_load_window_size = 2
    state.expert_load_window_step = 0
    state._slot_active = torch.zeros(2, dtype=torch.int32)
    state._slot_out_of_phase = torch.zeros(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((2,), -1)
    state._slot_gathered = None
    state.allgather_record_tensor = None
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._is_load_sampling_step = True
    state._load_collection_phase_matched = True
    for name, value in overrides.items():
        setattr(state, name, value)
    return state, upstream_step


def test_step_books_every_sampling_slot_and_writes_it(monkeypatch):
    # An idle rank whose batch is outside the phase still holds load sent by
    # the other ranks, so its slot is written like any other.
    state, upstream_step = _step_state(monkeypatch)
    state._load_collection_phase_matched = False

    state.step(is_dummy=True)
    state.expert_load_window_step = 1
    state._is_load_sampling_step = True
    state._load_collection_phase_matched = True
    state.step()

    assert upstream_step.call_args_list[0].kwargs["is_dummy"] is False
    assert upstream_step.call_args_list[1].kwargs["is_dummy"] is False
    torch.testing.assert_close(state._slot_active, torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(state._slot_out_of_phase, torch.tensor([0, 0], dtype=torch.int32))
    torch.testing.assert_close(state._physical_load_sample_slots, torch.tensor([0, 1]))
    assert state._num_recorded_load_steps == 2


def test_step_passes_non_sampling_steps_through(monkeypatch):
    state, upstream_step = _step_state(monkeypatch)
    state._is_load_sampling_step = False

    state.step(is_dummy=True)

    assert upstream_step.call_args.kwargs["is_dummy"] is True
    assert state._num_recorded_load_steps == 0
    assert bool((state._physical_load_sample_slots == -1).all())


class _NoHostReadTensor(torch.Tensor):
    """A scalar that fails the test when it is read on the host."""

    def item(self):
        raise AssertionError("the AllGather decision was read on the host during a step")


def test_step_copies_the_allgather_decision_without_a_host_read(monkeypatch):
    # The decision is a device scalar; reading it per step would sync the NPU.
    state, _ = _step_state(monkeypatch)
    state._slot_gathered = torch.full((2,), -1, dtype=torch.int32)
    state.allgather_record_tensor = torch.tensor(1, dtype=torch.int32).as_subclass(_NoHostReadTensor)

    state.step(is_dummy=True)

    assert state._slot_gathered.tolist() == [1, -1]


def test_mapping_change_discards_old_samples():
    state = AscendEplbState.__new__(AscendEplbState)
    model_state = _model_state(
        _load_mapping_generation=2,
        _observed_load_mapping_generation=1,
    )
    state.model_states = {"model": model_state}
    state._slot_active = torch.ones(3, dtype=torch.int32)
    state._slot_out_of_phase = torch.ones(3, dtype=torch.int32)
    state._physical_load_sample_slots = torch.arange(3)
    state._slot_gathered = torch.ones(3, dtype=torch.int32)
    state._num_recorded_load_steps = 3
    state._load_stats_window_start_index = 1
    state._load_stats_window_write_index = 2

    state._discard_samples_from_old_mapping()

    assert not state._slot_active.any()
    assert not state._slot_out_of_phase.any()
    assert bool((state._physical_load_sample_slots == -1).all())
    assert bool((state._slot_gathered == -1).all())
    assert state._num_recorded_load_steps == 0
    assert model_state._observed_load_mapping_generation == 2


def test_initialize_load_stats_adds_empty_commit_anchors():
    state = AscendEplbState.__new__(AscendEplbState)
    state.expert_load_window_size = 3
    state.policy = _custom_policy()
    model_state = _model_state(model=SimpleNamespace(num_moe_layers=2))

    state._initialize_load_stats_state(model_state)

    np.testing.assert_array_equal(
        np.isnan(model_state._last_committed_mean_ratios),
        [True, True],
    )


def test_collect_global_load_stats_maps_physical_to_logical(monkeypatch):
    ep_cpu_group = MagicMock()
    ep_cpu_group.size.return_value = 2
    monkeypatch.setattr(
        state_module,
        "get_ep_group",
        lambda: SimpleNamespace(cpu_group=ep_cpu_group, device_group=object()),
    )
    device_group = object()
    monkeypatch.setattr(
        state_module,
        "get_eplb_group",
        lambda: SimpleNamespace(cpu_group=object(), device_group=device_group),
    )
    all_reduce = MagicMock()
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)
    model_state = _model_state(
        model=SimpleNamespace(num_logical_experts=2),
        physical_to_logical_map=torch.tensor([[1, 0, 1]]),
        expert_load_window=torch.tensor([[[2, 3, 5]], [[7, 11, 13]], [[17, 19, 23]]]),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.device = torch.device("cpu")
    state.policy = _custom_policy()
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 3
    state._slot_active = torch.tensor([1, 0, 1], dtype=torch.int32)
    state._slot_out_of_phase = torch.zeros(3, dtype=torch.int32)
    state._physical_load_sample_slots = torch.tensor([0, -1, 2])
    state._num_recorded_load_steps = 3
    state._load_stats_window_start_index = 0

    result = state.collect_global_load_stats()

    assert result is not None
    torch.testing.assert_close(
        result["model"].values,
        torch.tensor([[[3, 7]], [[19, 40]]]),
    )
    # The slot flags are summed on the EP CPU group, the load on the EPLB device group.
    assert [call.kwargs["group"] for call in all_reduce.call_args_list] == [
        ep_cpu_group,
        device_group,
    ]


def test_maps_aggregated_physical_stats_to_logical():
    model_state = _model_state(
        model=SimpleNamespace(num_logical_experts=2),
        physical_to_logical_map=torch.tensor([[1, 0, 1]]),
    )

    result = AscendEplbState._map_physical_stats_to_logical(
        model_state,
        PreparedLoadStats(torch.tensor([[2, 3, 5]])),
    )

    torch.testing.assert_close(result.values, torch.tensor([[3, 7]]))


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
    model_state = _model_state(
        model=SimpleNamespace(
            num_physical_experts=8,
            num_expert_groups=1,
        ),
        rebalanced=False,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.get_rank_node_ids = MagicMock(return_value=np.array([0, 0, 1, 1]))
    state.rearrange_event = MagicMock()
    stats = PreparedLoadStats(torch.ones(2, 1, 3), np.array([2, 1]))

    state.publish_async_load_stats({"model": stats})

    assert model_state._policy_load_stats is stats
    assert model_state.eplb_stats.num_nodes == 2
    assert model_state.eplb_stats.num_gpus == 4
    assert model_state.rebalanced
    state.get_rank_node_ids.assert_called_once_with()
    state.rearrange_event.record.assert_called_once_with()


def test_rank_node_ids_are_discovered_once(monkeypatch):
    cpu_group = MagicMock()
    cpu_group.size.return_value = 5
    monkeypatch.setattr(
        state_module,
        "get_eplb_group",
        lambda: SimpleNamespace(cpu_group=cpu_group),
    )
    same_node = MagicMock(
        side_effect=(
            [True, False, True, False, False],
            [False, True, False, True, False],
            [False, False, False, False, True],
        )
    )
    monkeypatch.setattr(state_module, "in_the_same_node_as", same_node)
    state = AscendEplbState.__new__(AscendEplbState)

    first = state.get_rank_node_ids()
    second = state.get_rank_node_ids()

    np.testing.assert_array_equal(first, [0, 1, 0, 1, 2])
    np.testing.assert_array_equal(second, first)
    assert [call.args for call in same_node.call_args_list] == [
        (cpu_group, 0),
        (cpu_group, 1),
        (cpu_group, 4),
    ]
