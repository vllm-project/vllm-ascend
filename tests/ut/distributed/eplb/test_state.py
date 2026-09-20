# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb import state as eplb_state
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


def test_build_temporal_load_bins_orders_ring_and_preserves_counts():
    state = AscendEplbState.__new__(AscendEplbState)
    state.expert_load_window_size = 5
    state._logical_load_window_write_index = 2
    state._stair_config = StairConfig(load_window_bins=2)
    model_state = SimpleNamespace(
        _num_recorded_logical_load_samples=5,
        _logical_load_window=torch.tensor([[[4]], [[5]], [[1]], [[2]], [[3]]]),
    )

    included_sample_mask = torch.tensor([True, False, True, False, True])
    load_sums_per_bin, samples_per_bin = state._build_temporal_load_bins(model_state, included_sample_mask)

    torch.testing.assert_close(load_sums_per_bin, torch.tensor([[[1]], [[8]]]))
    assert samples_per_bin.tolist() == [1, 2]


def test_publish_temporal_load_stats_publishes_temporal_bins(monkeypatch):
    group = MagicMock()
    group.size.return_value = 2
    monkeypatch.setattr(eplb_state, "get_ep_group", lambda: SimpleNamespace(device_group=group))

    def simulate_all_reduce(tensor, **_):
        if tensor.ndim == 1:
            tensor[1] = 1

    all_reduce_mock = MagicMock(side_effect=simulate_all_reduce)
    monkeypatch.setattr(eplb_state, "all_reduce", all_reduce_mock)
    model_state = SimpleNamespace(
        model=SimpleNamespace(num_physical_experts=4, num_expert_groups=1),
        rebalanced=False,
        _num_recorded_logical_load_samples=2,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 2
    state._logical_load_window_write_index = 0
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state.rearrange_event = MagicMock()
    state._build_temporal_load_bins = MagicMock(
        return_value=(torch.tensor([[[1, 2]], [[3, 4]]]), torch.tensor([2, 3]).numpy())
    )

    state._publish_temporal_load_stats()

    torch.testing.assert_close(
        model_state.eplb_stats.global_expert_load_window,
        torch.tensor([[[1, 2]], [[3, 4]]]),
    )
    assert model_state._samples_per_load_bin.tolist() == [2, 3]
    assert model_state.eplb_stats.num_nodes == 1
    assert model_state.rebalanced
    torch.testing.assert_close(
        state._build_temporal_load_bins.call_args.args[1],
        torch.tensor([False, True]),
    )
    assert all_reduce_mock.call_count == 2
    all_reduce_mock.assert_called_with(model_state.eplb_stats.global_expert_load_window, group=group)
    state.rearrange_event.record.assert_called_once_with()


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
