# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state
from vllm.distributed.eplb.policy import DefaultEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb import state as eplb_state
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy
from vllm_ascend.distributed.eplb.state import (
    AscendEplbLayerState,
    AscendEplbState,
)


def test_uses_upstream_async_worker_lifecycle():
    assert AscendEplbState.start_async_loop is upstream_eplb_state.EplbState.start_async_loop


def test_step_records_logical_load_before_mapping_changes(monkeypatch):
    model_state = SimpleNamespace(
        _logical_load_window=torch.zeros((2, 1, 3), dtype=torch.int64),
        _num_recorded_logical_load_samples=0,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 2]]),
        expert_load_pass=torch.tensor([[2, 3, 5, 7]]),
    )
    new_mapping = torch.tensor([[2, 1, 0, 2]])
    upstream_step = MagicMock(side_effect=lambda **_: setattr(model_state, "physical_to_logical_map", new_mapping))
    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 2
    state._logical_load_window_write_index = 0
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.policy = StairEplbPolicy(StairConfig())

    state.step()
    model_state.expert_load_pass = torch.tensor([[11, 13, 17, 19]])
    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.step()

    torch.testing.assert_close(
        model_state._logical_load_window,
        torch.tensor([[[7, 3, 7]], [[17, 13, 30]]]),
    )
    assert model_state._num_recorded_logical_load_samples == 2
    assert state._logical_load_window_write_index == 0
    assert not state._is_load_sampling_step
    assert not state._should_collect_local_load
    assert upstream_step.call_count == 2


def test_collect_then_publish_async_load_stats(monkeypatch):
    group = MagicMock()
    group.size.return_value = 2
    ep_group = SimpleNamespace(device_group=group)
    monkeypatch.setattr(eplb_state, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(upstream_eplb_state, "get_ep_group", lambda: ep_group)

    def simulate_all_reduce(tensor, **_):
        if tensor.ndim == 1:
            tensor[::2] = 1
        else:
            tensor.mul_(2)

    all_reduce_mock = MagicMock(side_effect=simulate_all_reduce)
    monkeypatch.setattr(eplb_state, "all_reduce", all_reduce_mock)
    monkeypatch.setattr(upstream_eplb_state, "all_reduce", all_reduce_mock)
    model_state = SimpleNamespace(
        model=SimpleNamespace(num_physical_experts=4, num_expert_groups=1),
        rebalanced=False,
        _num_recorded_logical_load_samples=5,
        _logical_load_window=torch.tensor([[[4]], [[5]], [[1]], [[2]], [[3]]]),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 5
    state._logical_load_window_write_index = 2
    state._local_load_collection_mask = torch.zeros(5, dtype=torch.int32)
    state.get_rank_node_ids = MagicMock(return_value=np.array([0, 1]))
    state.rearrange_event = MagicMock()
    state.policy = StairEplbPolicy(StairConfig(load_window_bins=2))

    global_load_stats = state.collect_global_load_stats()
    state.publish_async_load_stats(global_load_stats)

    torch.testing.assert_close(
        model_state.eplb_stats.global_expert_load_window,
        torch.tensor([[[1]], [[8]]]) * 2,
    )
    assert model_state._policy_load_stats.sample_counts.tolist() == [1, 2]
    assert model_state.eplb_stats.num_nodes == 2
    assert model_state.rebalanced
    assert isinstance(global_load_stats["model"], PreparedLoadStats)
    assert all_reduce_mock.call_count == 2
    state.get_rank_node_ids.assert_called_once_with()
    state.rearrange_event.record.assert_called_once_with()


def test_rank_node_ids_are_discovered_once(monkeypatch):
    cpu_group = MagicMock()
    cpu_group.size.return_value = 5
    monkeypatch.setattr(eplb_state, "get_eplb_group", lambda: SimpleNamespace(cpu_group=cpu_group))
    same_node = MagicMock(
        side_effect=(
            [True, False, True, False, False],
            [False, True, False, True, False],
            [False, False, False, False, True],
        )
    )
    monkeypatch.setattr(eplb_state, "in_the_same_node_as", same_node)
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


def test_default_policy_uses_upstream_rearrange(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = DefaultEplbPolicy
    state.is_async = True
    state.parallel_config = SimpleNamespace(enable_elastic_ep=False)
    state._has_fresh_recorded_load = False
    state.model_states = {}
    upstream_rearrange = MagicMock(return_value=None)
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)

    state.rearrange()

    upstream_rearrange.assert_called_once_with(is_profile=False, rank_mapping=None)


def test_publish_requires_stats_for_every_model():
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": SimpleNamespace()}

    with pytest.raises(ValueError, match="exactly one entry"):
        state.publish_async_load_stats({})


def test_add_model_initializes_custom_load_history(monkeypatch):
    policy = StairEplbPolicy(StairConfig())
    model = SimpleNamespace(num_moe_layers=2, num_logical_experts=4)
    model_state = SimpleNamespace(model=model)
    model_config = SimpleNamespace(compute_hash=lambda: "model")
    state = AscendEplbState.__new__(AscendEplbState)
    state._configured_policy = policy
    state.model_states = {}
    state.expert_load_window_size = 3
    state.device = torch.device("cpu")

    def upstream_add_model(*_):
        state.policy = DefaultEplbPolicy
        state.model_states["model"] = model_state

    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", upstream_add_model)

    state.add_model(model, model_config)

    assert state.policy is policy
    assert model_state._logical_load_window.shape == (3, 2, 4)
    np.testing.assert_array_equal(np.isnan(model_state._last_committed_mean_ratios), [True, True])


def test_from_mapping_skips_custom_buffers_without_policy(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": SimpleNamespace()}
    state._initialize_load_stats_buffers = MagicMock()
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(lambda cls, **kwargs: state),
    )
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", MagicMock())

    AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros((1, 1)),
        policy=None,
    )

    state._initialize_load_stats_buffers.assert_not_called()


def test_layer_state_builds_routing_table_and_preserves_captured_tensor(
    monkeypatch,
):
    old_routing_table = torch.full((2, 2), -1, dtype=torch.int32)
    new_routing_table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    build_routing_table = MagicMock(side_effect=[old_routing_table, new_routing_table])
    monkeypatch.setattr(
        eplb_state,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    monkeypatch.setattr(
        eplb_state._eplb_ops,
        "build_expert_replica_routing_table",
        build_routing_table,
    )
    layer_state = AscendEplbLayerState()

    layer_state.set_layer_state(
        0,
        torch.zeros((1, 4), dtype=torch.int32),
        torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32),
        torch.tensor([[2, 2]], dtype=torch.int32),
    )
    captured_routing_table = layer_state.expert_replica_routing_table
    layer_state.refresh_expert_replica_routing_table()

    assert captured_routing_table is old_routing_table
    assert layer_state.expert_replica_routing_table is captured_routing_table
    torch.testing.assert_close(captured_routing_table, new_routing_table)


def test_sync_rearrange_refreshes_all_model_routing_tables(monkeypatch):
    sentinel = object()
    model_states = {"model": object()}

    def upstream_rearrange(self, is_profile=False, rank_mapping=None):
        assert not is_profile
        assert rank_mapping == {0: 0}
        return sentinel

    refresh = MagicMock()
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "rearrange",
        upstream_rearrange,
    )
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = False
    state.model_states = model_states

    result = state.rearrange(rank_mapping={0: 0})

    assert result is sentinel
    refresh.assert_called_once_with(model_states["model"])


def test_async_rearrange_defers_routing_refresh_to_workspace_hook(monkeypatch):
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "rearrange",
        lambda self, is_profile=False, rank_mapping=None: None,
    )
    refresh = MagicMock()
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = True
    state.model_states = {"model": object()}

    state.rearrange(rank_mapping={0: 0})

    refresh.assert_not_called()


def test_from_mapping_refreshes_final_mapping(monkeypatch):
    model_state = object()

    def upstream_from_mapping(cls, **kwargs):
        state = cls.__new__(cls)
        state.model_states = {"model": model_state}
        return state

    refresh = MagicMock()
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)

    state = AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros(1),
    )

    assert isinstance(state, AscendEplbState)
    refresh.assert_called_once_with(model_state)


def test_from_mapping_forwards_release_valid_expert_count(monkeypatch):
    received_count = None

    def upstream_from_mapping(
        cls,
        model,
        model_config,
        device,
        parallel_config,
        expanded_physical_to_logical,
        num_valid_physical_experts,
    ):
        del model, model_config, device, parallel_config
        del expanded_physical_to_logical
        nonlocal received_count
        received_count = num_valid_physical_experts
        state = cls.__new__(cls)
        state.model_states = {}
        return state

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )

    AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros((1, 2)),
        num_valid_physical_experts=1,
    )

    assert received_count == 1


def test_from_mapping_requires_release_valid_expert_count(monkeypatch):
    def upstream_from_mapping(
        cls,
        model,
        model_config,
        device,
        parallel_config,
        expanded_physical_to_logical,
        num_valid_physical_experts,
    ):
        raise AssertionError("release mapping must receive a valid count")

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )

    with pytest.raises(TypeError, match="required by the selected vLLM release"):
        AscendEplbState.from_mapping(
            model=object(),
            model_config=object(),
            device=torch.device("cpu"),
            parallel_config=object(),
            expanded_physical_to_logical=torch.zeros((1, 2)),
        )


def test_init_sets_cuda_device_index_for_npu(monkeypatch):
    parallel_config = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 5)
    monkeypatch.setattr(torch, "Event", torch.npu.Event)

    state = AscendEplbState(parallel_config, torch.device("cpu"))

    assert state.cuda_device_index == 5
