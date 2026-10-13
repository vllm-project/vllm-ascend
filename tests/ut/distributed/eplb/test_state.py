# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb import eplb_state
from vllm_ascend.distributed.eplb.eplb_communicator import AscendHixlEplbCommunicator, _HixlSession
from vllm_ascend.distributed.eplb.eplb_state import (
    AscendEplbLayerState,
    AscendEplbState,
)
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy
from vllm_ascend.utils import vllm_version_is


def test_uses_upstream_policy_and_async_worker_lifecycle():
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state._suspended = False
    state.model_states = {}
    with patch.object(upstream_eplb_state.EplbState, "start_async_loop") as start:
        state.start_async_loop()
    start.assert_called_once_with()


def test_result_readiness_defers_incomplete_transfer(monkeypatch):
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_state.get_ep_group",
        lambda: SimpleNamespace(cpu_group=group),
    )
    works = []

    def all_reduce(flag, *, group, async_op):
        assert group.size() == 2
        assert async_op
        if flag.item():
            flag.fill_(2)
        work = SimpleNamespace(wait=MagicMock())
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state.async_worker = None
    model_state = SimpleNamespace(pending_result=None)

    assert not state._all_ranks_result_ready(model_state)
    model_state.pending_result = object()
    assert not state._all_ranks_result_ready(model_state)
    assert state._all_ranks_result_ready(model_state)
    assert len(works) == 2
    for work in works:
        work.wait.assert_called_once_with()
    assert not hasattr(model_state, "_eplb_ready_work")
    assert not hasattr(model_state, "_eplb_ready_flag")
    assert model_state._eplb_foreground_wait_ms >= 0
    assert model_state._eplb_migration_span_steps == 3
    assert model_state._eplb_migration_deferred_steps == 2


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
        pending_result=SimpleNamespace(
            layer_idx=0,
            is_last_result=True,
            consumed_event=consumed_event,
        ),
    )
    state = SimpleNamespace(
        is_async=True,
        model_states={"model": model_state},
    )

    AscendEplbState.drain_async(state)

    assert not model_state.rebalanced
    assert model_state.pending_result is None
    consumed_event.record.assert_called_once_with()


def test_drain_async_fails_when_worker_stops():
    state = SimpleNamespace(
        is_async=True,
        async_worker=SimpleNamespace(is_alive=lambda: False),
        model_states={"model": SimpleNamespace(rebalanced=True, pending_result=None)},
    )

    with pytest.raises(RuntimeError, match="background worker terminated"):
        AscendEplbState.drain_async(state)


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
    state._hixl_session = None
    state._close_error = None
    state._suspended = False
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
    state._hixl_session = None
    state._close_error = None
    state._suspended = False
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
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(policy="default"),
        ),
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
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(policy="default"),
        ),
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
    monkeypatch.setattr(upstream_eplb_state, "CpuGpuEvent", MagicMock)

    state = AscendEplbState(parallel_config, torch.device("cpu"))

    if vllm_version_is("0.30.0"):
        assert state.cuda_device_index == 5
    else:
        assert state.device_index == 5


def _shutdown_state():
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state._suspended = False
    state._stop_async = threading.Event()
    state._close_error = None
    state.is_async = True
    state.async_worker = None
    state.model_states = {}
    return state


def test_close_wakes_idle_worker_without_fake_device_event():
    state = _shutdown_state()
    recorded = threading.Event()
    event_wait = MagicMock()
    state.rearrange_event = SimpleNamespace(_recorded=recorded, wait=event_wait)
    state.async_worker = threading.Thread(target=state.wait_for_rearrangement, args=(None,))
    state.async_worker.start()
    state.close()
    assert state.async_worker is None
    assert state._suspended
    assert not recorded.is_set()
    event_wait.assert_not_called()


def test_close_failure_keeps_communicator_owner():
    state = _shutdown_state()
    communicator = MagicMock(spec=AscendHixlEplbCommunicator)
    session = MagicMock(spec=_HixlSession)
    session._error = None
    session.close.side_effect = RuntimeError("still bound")
    state._hixl_session = session
    state.model_states = {"model": SimpleNamespace(communicator=communicator)}
    with pytest.raises(SystemExit, match="worker must terminate"):
        state.close()
    assert state.model_states["model"].communicator is communicator
    assert state._suspended


def test_failed_close_is_fatal_without_retrying_cleanup():
    state = _shutdown_state()
    error = RuntimeError("still bound")
    state._close_error = error
    communicator = MagicMock(spec=AscendHixlEplbCommunicator)
    state.model_states = {"model": SimpleNamespace(communicator=communicator)}
    with pytest.raises(SystemExit, match="worker must terminate") as exc:
        state.close()
    assert exc.value.__cause__ is error
    assert state._hixl_session is None


def test_hixl_initializes_all_models_before_starting_the_async_worker(monkeypatch):
    state = _shutdown_state()
    group = object()
    draft = AscendHixlEplbCommunicator.__new__(AscendHixlEplbCommunicator)
    main = AscendHixlEplbCommunicator.__new__(AscendHixlEplbCommunicator)
    draft._cpu_group = main._cpu_group = group
    state.model_states = {"draft": SimpleNamespace(communicator=draft), "main": SimpleNamespace(communicator=main)}
    session = MagicMock(spec=_HixlSession)
    session._error = None
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(eplb_state, "_HixlSession", factory)
    calls: list[tuple[str, list[str] | None]] = []
    session.initialize.side_effect = lambda models: calls.append(("initialize", list(models)))
    monkeypatch.setattr(upstream_eplb_state.EplbState, "start_async_loop", lambda *_args: calls.append(("start", None)))
    state.start_async_loop()
    state.start_async_loop()
    assert calls == [("initialize", ["draft", "main"]), ("start", None), ("start", None)]
    factory.assert_called_once_with(group)
    assert state._hixl_session is session
    with (
        patch.object(upstream_eplb_state.EplbState, "add_model") as add,
        pytest.raises(RuntimeError, match="Cannot add"),
    ):
        state.add_model(object(), object())
    add.assert_not_called()
    state.close()
    state.close()
    # Actual sessions perform teardown only once, even on repeated state close.
    assert session.close.call_count == 2


def test_hixl_initialization_failure_never_starts_the_async_worker(monkeypatch):
    state = _shutdown_state()
    main = AscendHixlEplbCommunicator.__new__(AscendHixlEplbCommunicator)
    main._cpu_group = object()
    state.model_states = {"main": SimpleNamespace(communicator=main)}
    session = MagicMock(spec=_HixlSession)
    session.initialize.side_effect = RuntimeError("registration failed")
    monkeypatch.setattr(eplb_state, "_HixlSession", MagicMock(return_value=session))
    with (
        patch.object(upstream_eplb_state.EplbState, "start_async_loop") as start,
        pytest.raises(RuntimeError, match="registration failed"),
    ):
        state.start_async_loop()
    start.assert_not_called()
    assert state._hixl_session is session


@pytest.mark.parametrize("entry", ["step", "ready", "close"])
def test_hixl_worker_setup_failure_exits_before_releasing_storage(entry):
    state = _shutdown_state()
    session = MagicMock(spec=_HixlSession)
    session._error = None
    session._usable = True
    state._hixl_session = session
    state.async_worker = SimpleNamespace(is_alive=lambda: False)
    state.model_states = {"main": SimpleNamespace(rebalanced=False, pending_result=None)}
    with pytest.raises(SystemExit, match="worker must terminate") as fatal:
        if entry == "step":
            state.step()
        elif entry == "ready":
            state._all_ranks_result_ready(object())
        else:
            state.close()
    assert "background worker terminated" in str(fatal.value.__cause__)
    assert session._error is fatal.value.__cause__
    assert not session._usable
    session.close.assert_not_called()


@pytest.mark.parametrize("entry", ["step", "ready", "close"])
def test_failed_shared_session_exits_from_foreground_before_reusing_resources(entry):
    state = _shutdown_state()
    error = RuntimeError("HIXL transfer failed")
    session = MagicMock(spec=_HixlSession)
    session._error = error
    state._hixl_session = session
    with pytest.raises(SystemExit, match="worker must terminate") as fatal:
        if entry == "step":
            state.step()
        elif entry == "ready":
            state._all_ranks_result_ready(object())
        else:
            state.close()
    assert fatal.value.__cause__ is error
    assert state._hixl_session is session
    session.close.assert_not_called()
