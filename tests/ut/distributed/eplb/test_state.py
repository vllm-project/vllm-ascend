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


def test_configured_upstream_policy_registration_is_scoped():
    policy = StairEplbPolicy(StairConfig())
    assert "stair" not in upstream_eplb_state.EPLB_POLICIES

    with eplb_state._configured_upstream_policy("stair", policy):
        assert upstream_eplb_state.EPLB_POLICIES["stair"] is policy

    assert "stair" not in upstream_eplb_state.EPLB_POLICIES


def test_drain_async_accepts_last_changed_layer_before_model_end():
    consumed_event = MagicMock()
    model_state = SimpleNamespace(
        rebalanced=True,
        model=SimpleNamespace(num_moe_layers=3),
        pending_result=SimpleNamespace(layer_idx=0, is_last_result=True, consumed_event=consumed_event),
    )
    state = SimpleNamespace(
        is_async=True,
        model_states={"model": model_state},
        _all_ranks_result_ready=lambda _model_state: True,
    )

    AscendEplbState.drain_async(state)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    consumed_event.record.assert_called_once_with()


def test_step_discards_physical_samples_after_mapping_changes(monkeypatch):
    model_state = SimpleNamespace(
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 2]]),
        expert_load_pass=torch.tensor([[2, 3, 5, 7]]),
        expert_load_window=torch.zeros((2, 1, 4), dtype=torch.int32),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 2
    state.expert_load_window_step = 0
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((2,), -1, dtype=torch.long)
    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.policy = StairEplbPolicy(StairConfig())

    def upstream_step(_self, **_):
        write_index = state.expert_load_window_step
        model_state.expert_load_window[write_index].copy_(model_state.expert_load_pass)
        state.expert_load_window_step = (write_index + 1) % state.expert_load_window_size

    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)
    state.step()
    model_state._load_mapping_generation += 1
    model_state.expert_load_pass = torch.tensor([[11, 13, 17, 19]])
    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.step()

    assert state._num_recorded_load_steps == 1
    assert state._load_stats_window_start_index == 0
    torch.testing.assert_close(state._local_load_collection_mask, torch.tensor([1, 0], dtype=torch.int32))
    torch.testing.assert_close(state._physical_load_sample_slots, torch.tensor([1, -1]))
    torch.testing.assert_close(
        model_state.expert_load_window[1],
        torch.tensor([[11, 13, 17, 19]], dtype=torch.int32),
    )
    assert model_state._observed_load_mapping_generation == 1
    assert not state._is_load_sampling_step
    assert not state._should_collect_local_load


@pytest.mark.parametrize("is_dummy", [False, True])
def test_step_keeps_uncollected_load_slot_aligned(monkeypatch, is_dummy):
    model_state = SimpleNamespace(
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 2
    state.expert_load_window_step = 0
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((2,), -1, dtype=torch.long)
    state._is_load_sampling_step = not is_dummy
    state._should_collect_local_load = False
    state.policy = StairEplbPolicy(StairConfig())

    def upstream_step(_self, *, is_dummy=False, **_):
        if not is_dummy:
            state.expert_load_window_step = (state.expert_load_window_step + 1) % state.expert_load_window_size

    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)

    state.step(is_dummy=is_dummy)

    assert state._num_recorded_load_steps == 1
    torch.testing.assert_close(state._local_load_collection_mask, torch.zeros(2, dtype=torch.int32))

    state._is_load_sampling_step = True
    state._should_collect_local_load = True
    state.step()

    assert state._num_recorded_load_steps == 2
    torch.testing.assert_close(state._local_load_collection_mask, torch.tensor([0, 1], dtype=torch.int32))
    expected_physical_slot = 1 if not is_dummy else 0
    torch.testing.assert_close(
        state._physical_load_sample_slots,
        torch.tensor([-1, expected_physical_slot]),
    )


def test_physical_stats_are_mapped_after_binning():
    model_state = SimpleNamespace(
        model=SimpleNamespace(num_logical_experts=2),
        physical_to_logical_map=torch.tensor([[0, 1, 0, -1]]),
    )
    physical_stats = PreparedLoadStats(
        torch.tensor([[[2, 3, 5, 7]], [[11, 13, 17, 19]]]),
        np.array([1, 2]),
    )

    logical_stats = AscendEplbState._map_physical_stats_to_logical(model_state, physical_stats)

    torch.testing.assert_close(logical_stats.values, torch.tensor([[[7, 3]], [[28, 13]]]))
    np.testing.assert_array_equal(logical_stats.sample_counts, [1, 2])


def test_collect_discards_window_committed_under_old_mapping():
    model_state = SimpleNamespace(
        _load_mapping_generation=1,
        _observed_load_mapping_generation=0,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.policy = StairEplbPolicy(StairConfig())
    state.expert_load_window_size = 2
    state.expert_load_window_step = 1
    state._num_recorded_load_steps = 2
    state._load_stats_window_start_index = 1
    state._load_stats_window_write_index = 1
    state._local_load_collection_mask = torch.ones(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.arange(2)

    assert state.collect_global_load_stats() is None
    assert state._num_recorded_load_steps == 0
    assert state._load_stats_window_start_index == 0
    assert state._load_stats_window_write_index == 0
    torch.testing.assert_close(state._local_load_collection_mask, torch.zeros(2, dtype=torch.int32))
    torch.testing.assert_close(state._physical_load_sample_slots, torch.full((2,), -1))


def test_collect_then_publish_async_load_stats(monkeypatch):
    device_group = MagicMock()
    device_group.size.return_value = 2
    cpu_group = MagicMock()
    ep_group = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)
    monkeypatch.setattr(eplb_state, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(eplb_state, "get_eplb_group", lambda: ep_group)
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
        model=SimpleNamespace(num_physical_experts=1, num_logical_experts=1, num_expert_groups=1),
        rebalanced=False,
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
        physical_to_logical_map=torch.tensor([[0]]),
        expert_load_window=torch.tensor([[[4]], [[5]], [[1]], [[2]], [[3]]], dtype=torch.int32),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 5
    state._num_recorded_load_steps = 5
    state._load_stats_window_start_index = 2
    state._load_stats_window_write_index = 2
    state._local_load_collection_mask = torch.zeros(5, dtype=torch.int32)
    state._physical_load_sample_slots = torch.arange(5)
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


def test_async_rearrange_skips_window_without_collected_load(monkeypatch):
    group = MagicMock()
    monkeypatch.setattr(
        eplb_state,
        "get_eplb_group",
        lambda: SimpleNamespace(device_group=group, cpu_group=group),
    )
    all_reduce = MagicMock()
    monkeypatch.setattr(eplb_state, "all_reduce", all_reduce)
    model_state = SimpleNamespace(
        model=SimpleNamespace(num_logical_experts=1),
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
        physical_to_logical_map=torch.tensor([[0]]),
        expert_load_window=torch.zeros((3, 1, 1), dtype=torch.int32),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = True
    state.parallel_config = SimpleNamespace(enable_elastic_ep=False)
    state.policy = StairEplbPolicy(StairConfig())
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 3
    state._num_recorded_load_steps = 3
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._local_load_collection_mask = torch.zeros(3, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((3,), -1, dtype=torch.long)
    state._has_fresh_recorded_load = True  # An earlier sample was overwritten.
    state._has_global_fresh_recorded_load = MagicMock(return_value=True)
    state.publish_async_load_stats = MagicMock()

    assert state.rearrange() is None
    state.publish_async_load_stats.assert_not_called()
    assert not state._has_fresh_recorded_load
    assert all_reduce.call_count == 1
    torch.testing.assert_close(all_reduce.call_args.args[0], torch.zeros(3, dtype=torch.int32))
    assert all_reduce.call_args.kwargs["group"] is group

    model_state.expert_load_window = torch.tensor([[[2]], [[0]], [[0]]], dtype=torch.int32)
    state._local_load_collection_mask[0] = 1
    state._physical_load_sample_slots[0] = 0
    state._has_fresh_recorded_load = True

    state.rearrange()

    state.publish_async_load_stats.assert_called_once()
    assert not state._has_fresh_recorded_load


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


def test_add_model_initializes_custom_load_state(monkeypatch):
    policy = StairEplbPolicy(StairConfig())
    model = SimpleNamespace(num_moe_layers=2, num_logical_experts=4)
    model_state = SimpleNamespace(model=model)
    model_config = SimpleNamespace(compute_hash=lambda: "model")
    state = AscendEplbState.__new__(AscendEplbState)
    state._configured_policy = policy
    state.model_states = {}
    state.expert_load_window_size = 3
    state.expert_load_window_step = 0
    state.device = torch.device("cpu")

    def upstream_add_model(*_):
        state.policy = DefaultEplbPolicy
        state.model_states["model"] = model_state

    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", upstream_add_model)
    monkeypatch.setattr(eplb_state, "get_ep_group", lambda: SimpleNamespace(world_size=1))

    state.add_model(model, model_config)

    assert state.policy is policy
    assert state._local_load_collection_mask.device.type == "cpu"
    assert state._local_load_collection_mask.shape == (3,)
    assert state._num_recorded_load_steps == 0
    assert state._physical_load_sample_slots.shape == (3,)
    assert not hasattr(model_state, "_logical_load_window")
    np.testing.assert_array_equal(np.isnan(model_state._last_committed_mean_ratios), [True, True])


def test_from_mapping_skips_custom_buffers_without_policy(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = DefaultEplbPolicy
    state.model_states = {"model": SimpleNamespace()}
    state._initialize_load_stats_state = MagicMock()
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

    state._initialize_load_stats_state.assert_not_called()


def test_layer_state_builds_routing_table_and_preserves_captured_tensor(
    monkeypatch,
):
    old_routing_table = torch.full((2, 2), -1, dtype=torch.int32)
    new_routing_table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    build_routing_table = MagicMock(side_effect=[old_routing_table, new_routing_table])
    monkeypatch.setattr(
        eplb_state,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2),
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
        state.policy = DefaultEplbPolicy
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
        state.policy = DefaultEplbPolicy
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
    monkeypatch.setattr(torch.cuda, "Event", torch.npu.Event)

    state = AscendEplbState(parallel_config, torch.device("cpu"))

    assert state.cuda_device_index == 5
