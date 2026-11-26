# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state
from vllm.distributed.eplb.policy import DefaultEplbPolicy

from vllm_ascend.ascend_config import EplbConfig, StairConfig
from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.distributed.eplb.policy.factory import AscendDefaultEplbPolicy, create_eplb_policy
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy
from vllm_ascend.worker.v2.eplb import (
    AscendEPLBController,
    is_eplb_load_collection_phase_matched,
)


class TestEplbLoadCollectionPhase(unittest.TestCase):
    def test_load_collection_phase_semantics(self):
        cases = [
            ("all", [True, False], True),
            ("all", [False, False], True),
            ("prefill", [True, False], True),
            ("decode", [True, False], False),
            ("prefill", [False, False], False),
            ("decode", [False, False], True),
        ]
        for load_collection_phase, is_prefilling, expected in cases:
            with self.subTest(
                load_collection_phase=load_collection_phase,
                is_prefilling=is_prefilling,
            ):
                self.assertIs(
                    is_eplb_load_collection_phase_matched(
                        load_collection_phase,
                        any(is_prefilling),
                    ),
                    expected,
                )

    @staticmethod
    def _make_controller(load_collection_phase="all", log_balancedness=False, policy="stair"):
        parallel_config = SimpleNamespace(
            enable_eplb=True,
            # Explicit communicator skips the group-wide consensus, which
            # needs a live EPLB group that unit tests do not have.
            eplb_config=SimpleNamespace(
                log_balancedness=log_balancedness,
                policy=policy,
                communicator="torch_gloo",
            ),
        )
        controller = AscendEPLBController(
            parallel_config,
            torch.device("cpu"),
            ascend_eplb_config=EplbConfig(
                load_collection_phase=load_collection_phase,
            ),
        )
        controller._has_registered_models = True
        return controller

    def test_prepare_load_constructs_ascend_state(self):
        controller = self._make_controller()

        with (
            patch("vllm.distributed.eplb.eplb_state.CpuGpuEvent"),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            controller.prepare_load()

        self.assertIsInstance(controller.state, AscendEplbState)
        self.assertIs(controller.state.policy, controller.eplb_policy)

    def test_policy_selection_uses_upstream_config(self):
        default_controller = self._make_controller(policy="default")
        stair_controller = self._make_controller(policy="stair")

        self.assertIsInstance(default_controller.eplb_policy, DefaultEplbPolicy)
        self.assertIsInstance(default_controller.eplb_policy, AscendDefaultEplbPolicy)
        self.assertIsInstance(stair_controller.eplb_policy, StairEplbPolicy)

    def test_setup_from_mapping_uses_current_upstream_contract(self):
        controller = self._make_controller()
        model_config = object()
        mapping = torch.zeros((1, 1), dtype=torch.int32)
        state = MagicMock()

        controller.state = state
        with patch.object(AscendEplbState, "from_mapping") as from_mapping:
            controller.setup_from_mapping(model_config, mapping)
        state.update_mapping.assert_called_once_with(model_config, mapping)
        from_mapping.assert_not_called()
        self.assertIs(controller.state, state)
        self.assertTrue(controller._has_registered_models)

    def test_setup_from_mapping_preserves_registered_state(self):
        controller = self._make_controller()
        state = MagicMock()
        controller.state = state
        controller._has_registered_models = True
        model_config = object()
        mapping = torch.zeros((1, 2), dtype=torch.int32)
        with patch.object(AscendEplbState, "from_mapping") as from_mapping:
            controller.setup_from_mapping(model_config, mapping)
        state.update_mapping.assert_called_once_with(model_config, mapping)
        from_mapping.assert_not_called()
        self.assertIs(controller.state, state)
        self.assertTrue(controller._has_registered_models)

    def test_prepare_forward_publishes_phase_match_without_gating_recording(self):
        for batch_has_prefill in (False, True):
            with self.subTest(batch_has_prefill=batch_has_prefill):
                controller = self._make_controller(load_collection_phase="prefill")
                state = MagicMock()
                state.should_record_tensor = torch.tensor(False)
                state._should_record_current_step.return_value = True
                controller.state = state
                controller.set_batch_phase(batch_has_prefill=batch_has_prefill)

                controller.prepare_forward(object(), 7)

                state.prepare_forward.assert_not_called()
                state._should_record_current_step.assert_called_once_with(log_stats=False)
                # The executing rank's counts are recorded; a slot is kept or
                # dropped when rearranging.
                self.assertTrue(state.should_record_tensor.item())
                self.assertTrue(state._is_load_sampling_step)
                self.assertIs(state._load_collection_phase_matched, batch_has_prefill)

    def test_default_policy_treats_the_phase_as_all(self):
        controller = self._make_controller(load_collection_phase="prefill", policy="default")
        state = MagicMock()
        state.uses_custom_load_stats = False
        state.should_record_tensor = torch.tensor(False)
        state._should_record_current_step.return_value = True
        controller.state = state
        controller.set_batch_phase(batch_has_prefill=False)

        controller.prepare_forward(object(), 7)

        self.assertTrue(state._load_collection_phase_matched)
        self.assertTrue(state.should_record_tensor.item())
        self.assertTrue(state._is_load_sampling_step)
        state.prepare_forward.assert_called_once()


_WINDOW_SIZE = 2


def _stair_policy():
    return SimpleNamespace(prepare_local_load_stats=lambda samples: PreparedLoadStats(samples))


_BOTH_POLICIES = pytest.mark.parametrize(
    "policy_factory",
    [_stair_policy, AscendDefaultEplbPolicy],
    ids=["stair", "default"],
)


def _model_state(window):
    return SimpleNamespace(
        model=SimpleNamespace(num_logical_experts=window.shape[-1]),
        physical_to_logical_map=torch.arange(window.shape[-1]).unsqueeze(0),
        _load_mapping_generation=0,
        _observed_load_mapping_generation=0,
        expert_load_window=window,
        expert_load_pass=torch.zeros_like(window[0]),
        rebalanced=False,
    )


def _install_groups(monkeypatch, remote=()):
    """Give the EP group CPU and device subgroups and mock the reduction.

    ``remote`` holds the per-slot ``[active, out_of_phase]`` that other ranks
    contribute to the window flags; other tensors are left as they are.
    """
    cpu_group = MagicMock()
    cpu_group.size.return_value = 4
    device_group = MagicMock()
    device_group.rank.return_value = 0
    reduced = []

    def all_reduce(value, **kwargs):
        reduced.append(kwargs["group"])
        if kwargs["group"] is cpu_group and value.shape == (2, _WINDOW_SIZE):
            for flags in remote:
                value.add_(torch.tensor(flags, dtype=value.dtype))

    ep_group = SimpleNamespace(cpu_group=cpu_group, device_group=device_group)
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)
    monkeypatch.setattr(state_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(state_module, "get_eplb_group", lambda: SimpleNamespace(device_group=device_group))
    return ep_group, reduced


def _booked_state(policy, *, active, out_of_phase, rows=None, gathered=None):
    """A state at rearrangement time; its window has one row per booked slot."""
    window = torch.zeros((_WINDOW_SIZE, 1, 2), dtype=torch.int32)
    if rows is not None:
        window.copy_(torch.tensor(rows, dtype=torch.int32))
    state = AscendEplbState.__new__(AscendEplbState)
    state._suspended = False
    state.parallel_config = SimpleNamespace(enable_elastic_ep=False)
    state.device = torch.device("cpu")
    state.policy = policy
    state.model_states = {"model": _model_state(window)}
    state.is_async = True
    state.expert_load_window_size = _WINDOW_SIZE
    state._slot_active = torch.tensor(active, dtype=torch.int32)
    state._slot_out_of_phase = torch.tensor(out_of_phase, dtype=torch.int32)
    state._physical_load_sample_slots = torch.arange(_WINDOW_SIZE)
    state._slot_gathered = None if gathered is None else torch.tensor(gathered, dtype=torch.int32)
    state._num_recorded_load_steps = _WINDOW_SIZE
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    return state


def _rearranged(policy, state, upstream_rearrange, publish):
    """Return the slot rows the policy was offered, or None when it was not."""
    if isinstance(policy, DefaultEplbPolicy):
        if not upstream_rearrange.called:
            return None
        window = state.model_states["model"].expert_load_window
        return window[window.abs().sum(dim=(1, 2)) > 0].tolist()
    if not publish.called:
        return None
    return publish.call_args.args[0]["model"].values.tolist()


@_BOTH_POLICIES
def test_slot_holding_load_of_an_idle_rank_is_rearranged(monkeypatch, policy_factory):
    # The three idle DP replicas of a group execute most of the tokens, so a
    # slot of a rank that scheduled nothing still holds load.
    ep_group, reduced = _install_groups(monkeypatch, remote=[([1, 1], [0, 0])])
    state = _booked_state(policy_factory(), active=[0, 0], out_of_phase=[0, 0], rows=[[[5, 7]], [[9, 11]]])
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(AscendEplbState, "publish_async_load_stats", publish)

    state.rearrange()

    assert _rearranged(state.policy, state, upstream_rearrange, publish) == [[[5, 7]], [[9, 11]]]
    # The window flags are reduced once per rearrangement on the EP CPU group.
    # STAIR then sums the selected samples on the EP device group.
    later = [] if isinstance(state.policy, DefaultEplbPolicy) else [ep_group.device_group]
    assert reduced == [ep_group.cpu_group] + later


@_BOTH_POLICIES
def test_slot_mixing_phases_across_ranks_is_dropped(monkeypatch, policy_factory):
    # Owner counts cannot tell which DP rank sent a token, so a slot in which
    # another rank prefills is not a decode sample.
    _install_groups(monkeypatch, remote=[([1, 0], [0, 0]), ([0, 1], [0, 1])])
    state = _booked_state(policy_factory(), active=[1, 1], out_of_phase=[0, 0], rows=[[[5, 7]], [[9, 11]]])
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(AscendEplbState, "publish_async_load_stats", publish)

    state.rearrange()

    assert _rearranged(state.policy, state, upstream_rearrange, publish) == [[[5, 7]]]


@_BOTH_POLICIES
def test_window_without_a_complete_slot_does_not_rearrange(monkeypatch, policy_factory):
    _install_groups(monkeypatch, remote=[([1, 0], [1, 1])])
    state = _booked_state(policy_factory(), active=[1, 1], out_of_phase=[1, 1], rows=[[[5, 7]], [[9, 11]]])
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(AscendEplbState, "publish_async_load_stats", publish)

    state.rearrange()

    assert _rearranged(state.policy, state, upstream_rearrange, publish) is None


def test_default_policy_preserves_placement_on_zero_load_layers():
    # Upstream DefaultEplbPolicy piles every redundant replica onto expert 0
    # when a layer's load is all zeros; AscendDefaultEplbPolicy keeps the
    # current placement on CPU without requiring a device-window reduction.
    policy = create_eplb_policy("default", StairConfig())
    old_map = torch.tensor([[0, 1, 2, 3, 2, 3], [0, 1, 2, 3, 1, 2]], dtype=torch.int32)

    all_zero = policy.rebalance_experts(
        torch.zeros((2, 4), dtype=torch.int64),
        num_replicas=6,
        num_groups=2,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_map,
    )
    assert all_zero.device.type == "cpu"
    torch.testing.assert_close(all_zero, old_map.to(torch.long))

    mixed_weight = torch.tensor([[0, 0, 0, 0], [100, 1, 1, 1]], dtype=torch.int64)
    mixed = policy.rebalance_experts(
        mixed_weight,
        num_replicas=6,
        num_groups=2,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=old_map,
    )
    torch.testing.assert_close(mixed[0], old_map[0].to(torch.long))
    assert int((mixed[1] == 0).sum()) > 1


def test_slot_without_an_active_rank_is_dropped_with_its_stale_load(monkeypatch):
    _install_groups(monkeypatch)
    state = _booked_state(DefaultEplbPolicy(), active=[0, 0], out_of_phase=[0, 0], rows=[[[5, 7]], [[9, 11]]])
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)

    state.rearrange()

    assert _rearranged(state.policy, state, upstream_rearrange, publish) is None


@pytest.mark.parametrize(
    "gathered,expected_rows",
    [
        # This rank scheduled nothing, but a source rank was inside the phase.
        ([1, -1], [[[5, 7]]]),
        # The sources filter by their own phase, which is finer than the slot flags.
        ([-1, 1], [[[9, 11]]]),
        ([-1, -1], None),
    ],
)
def test_allgather_slots_follow_the_gathered_sources(monkeypatch, gathered, expected_rows):
    _install_groups(monkeypatch)
    # Slot 0 has no active rank, slot 1 mixes phases: the flags would drop both.
    state = _booked_state(
        _stair_policy(),
        active=[0, 1],
        out_of_phase=[0, 1],
        rows=[[[5, 7]], [[9, 11]]],
        gathered=gathered,
    )
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(AscendEplbState, "publish_async_load_stats", publish)

    state.rearrange()

    assert _rearranged(state.policy, state, upstream_rearrange, publish) == expected_rows


@pytest.mark.parametrize("is_profile,enable_elastic_ep", [(True, False), (False, True)])
def test_profile_and_elastic_rearranges_bypass_the_gate(monkeypatch, is_profile, enable_elastic_ep):
    ep_group, reduced = _install_groups(monkeypatch)
    state = _booked_state(DefaultEplbPolicy(), active=[0, 0], out_of_phase=[0, 0])
    state.parallel_config.enable_elastic_ep = enable_elastic_ep
    upstream_rearrange = MagicMock(return_value=None)
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)

    state.rearrange(is_profile=is_profile)

    upstream_rearrange.assert_called_once_with(is_profile=is_profile, rank_mapping=None)
    assert reduced == []


@pytest.mark.parametrize("mixed,summary", [(False, "2 slots collected, 0 slots"), (True, "1 slots collected, 1 slots")])
def test_window_selection_is_logged_once_per_rearrangement(monkeypatch, mixed, summary):
    _install_groups(monkeypatch)
    state = _booked_state(DefaultEplbPolicy(), active=[1, 1], out_of_phase=[0, int(mixed)], rows=[[[5, 7]], [[9, 11]]])
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", MagicMock(return_value=None))
    logger = MagicMock()
    monkeypatch.setattr(state_module, "logger", logger)

    state.rearrange()

    assert logger.info.call_count == 1
    message, *counts = logger.info.call_args.args
    assert summary in message % tuple(counts)


def test_window_flags_fall_back_to_the_device_group(monkeypatch):
    # A group without a CPU subgroup reduces on the device one instead.
    device_group = MagicMock()
    device_group.size.return_value = 4
    monkeypatch.setattr(state_module, "get_ep_group", lambda: SimpleNamespace(device_group=device_group))
    all_reduce = MagicMock(side_effect=lambda value, **_kwargs: value.add_(3))
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)
    state = AscendEplbState.__new__(AscendEplbState)
    state.device = torch.device("cpu")
    flags = torch.tensor([1, 0, 5])

    state._all_reduce_sum(flags)

    assert flags.tolist() == [4, 3, 8]
    assert all_reduce.call_args.kwargs["group"] is device_group


def _live_state(monkeypatch, policy):
    """A state whose upstream step runs and rearranges every second step."""
    _install_groups(monkeypatch)
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: SimpleNamespace(device_group=MagicMock()),
    )
    upstream_rearrange = MagicMock(return_value=None)
    publish = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(AscendEplbState, "publish_async_load_stats", publish)
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state._suspended = False
    state.parallel_config = SimpleNamespace(enable_elastic_ep=False)
    state.device = torch.device("cpu")
    state.policy = policy
    state.model_states = {"model": _model_state(torch.zeros((_WINDOW_SIZE, 1, 2), dtype=torch.int32))}
    state.is_async = True
    state.expert_load_window_size = _WINDOW_SIZE
    state.expert_load_window_step = 0
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = _WINDOW_SIZE
    state.should_record_tensor = None
    state._slot_active = torch.zeros(_WINDOW_SIZE, dtype=torch.int32)
    state._slot_out_of_phase = torch.zeros(_WINDOW_SIZE, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((_WINDOW_SIZE,), -1)
    state._slot_gathered = None
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    state._is_load_sampling_step = False
    state._load_collection_phase_matched = True
    return state, upstream_rearrange, publish


def _record(state, load, *, is_dummy=False, phase_matched=True):
    """Run one sampling step as if the forward had recorded ``load``."""
    model_state = state.model_states["model"]
    model_state.expert_load_pass.fill_(0)
    model_state.expert_load_pass[0] = torch.tensor(load)
    state._is_load_sampling_step = True
    state._load_collection_phase_matched = phase_matched
    state.step(is_dummy=is_dummy)


@_BOTH_POLICIES
def test_interval_of_skipped_slots_restarts_the_clock_without_rearranging(monkeypatch, policy_factory):
    state, upstream_rearrange, publish = _live_state(monkeypatch, policy_factory())
    # Under a large DP another rank is likely inside the phase. That does not
    # make this rank's out-of-phase slots complete.
    _install_groups(monkeypatch, remote=[([1, 0], [0, 0])])

    for _ in range(4):
        _record(state, [5, 7], phase_matched=False)

    assert _rearranged(state.policy, state, upstream_rearrange, publish) is None
    # Upstream restarts its rearrangement clock whether or not a rearrangement ran.
    assert state.expert_rearrangement_step == 0


@_BOTH_POLICIES
@pytest.mark.parametrize("committed", [False, True], ids=["same_placement", "after_commit"])
def test_window_written_before_a_commit_does_not_rearrange(monkeypatch, policy_factory, committed):
    state, upstream_rearrange, publish = _live_state(monkeypatch, policy_factory())
    _record(state, [5, 7])
    if committed:
        # The commit of the last changed layer in refresh_model_routing_tables.
        state.model_states["model"].expert_load_window.zero_()
        state.model_states["model"]._load_mapping_generation += 1
    # An interval in which no rank schedules tokens.
    _record(state, [0, 0], is_dummy=True)

    offered = _rearranged(state.policy, state, upstream_rearrange, publish)
    assert offered == (None if committed else [[[5, 7]]])
