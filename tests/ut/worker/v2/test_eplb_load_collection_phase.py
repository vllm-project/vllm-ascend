# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state
from vllm.distributed.eplb.policy import DefaultEplbPolicy

from vllm_ascend.ascend_config import EplbConfig
from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbState
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
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
                state._has_fresh_recorded_load = False
                state._should_record_current_step.return_value = True
                controller.state = state
                controller.set_batch_phase(batch_has_prefill=batch_has_prefill)

                controller.prepare_forward(object(), 7)

                state.prepare_forward.assert_not_called()
                state._should_record_current_step.assert_called_once_with(log_stats=False)
                # The executing rank's counts are recorded; the step decides group-wide.
                self.assertTrue(state.should_record_tensor.item())
                self.assertTrue(state._is_load_sampling_step)
                self.assertIs(state._load_collection_phase_matched, batch_has_prefill)
                self.assertFalse(state._has_fresh_recorded_load)


def _reduced_by(rank_flags, sent):
    """Return an all_reduce that records the input and sums ``rank_flags`` into it."""

    def all_reduce(flags, **_kwargs):
        sent.append(flags.tolist())
        flags.copy_(torch.tensor(rank_flags, dtype=torch.int32).sum(dim=0))

    return all_reduce


def _decision_state(monkeypatch, *, phase_matched=True, gathered=None, shared_ep_group=True):
    monkeypatch.setattr(state_module, "_needs_group_load_decision", lambda: shared_ep_group)
    cpu_group = MagicMock()
    cpu_group.size.return_value = 4
    monkeypatch.setattr(state_module, "get_ep_group", lambda: SimpleNamespace(cpu_group=cpu_group))
    upstream_step = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "step", upstream_step)
    state = AscendEplbState.__new__(AscendEplbState)
    state._hixl_session = None
    state._close_error = None
    state._has_fresh_recorded_load = False
    state._is_load_sampling_step = True
    state._load_collection_phase_matched = phase_matched
    state.allgather_record_tensor = None if gathered is None else torch.tensor(gathered)
    state.policy = SimpleNamespace(prepare_local_load_stats=lambda samples: PreparedLoadStats(samples))
    state.model_states = {}
    state.expert_load_window_size = 2
    state.expert_load_window_step = 3
    state._local_load_collection_mask = torch.zeros(2, dtype=torch.int32)
    state._physical_load_sample_slots = torch.full((2,), -1)
    state._num_recorded_load_steps = 0
    state._load_stats_window_start_index = 0
    state._load_stats_window_write_index = 0
    return state, upstream_step, cpu_group


def _assert_committed(state, upstream_step, committed):
    # Committing passes ``is_dummy=False`` upstream so every rank keeps its window slot.
    assert upstream_step.call_args.kwargs["is_dummy"] == (not committed)
    assert int(state._local_load_collection_mask[0]) == int(committed)
    assert int(state._physical_load_sample_slots[0]) == (3 if committed else -1)
    assert state._has_fresh_recorded_load == committed


@pytest.mark.parametrize(
    "rank_flags,expected",
    [
        # An idle rank executes tokens sent by the active rank.
        ([(0, 0), (1, 0)], True),
        # Nobody schedules tokens.
        ([(0, 0), (0, 0)], False),
        # Every active rank is outside the collection phase.
        ([(0, 0), (1, 1)], False),
    ],
)
def test_idle_rank_follows_the_group(monkeypatch, rank_flags, expected):
    state, upstream_step, cpu_group = _decision_state(monkeypatch)
    sent = []
    all_reduce = MagicMock(side_effect=_reduced_by(rank_flags, sent))
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)

    state.step(is_dummy=True)

    # An idle rank never asks for the step by itself.
    assert sent == [[0, 0]]
    assert all_reduce.call_args.kwargs["group"] is cpu_group
    _assert_committed(state, upstream_step, expected)


@pytest.mark.parametrize(
    "phase_matched,rank_flags,expected",
    [
        # Every active rank is inside the collection phase.
        (True, [(1, 0), (1, 0), (0, 0)], True),
        # Another DP rank prefills while this one decodes: owner counts cannot split them.
        (True, [(1, 0), (1, 1), (0, 0)], False),
        (False, [(1, 1), (1, 0), (0, 0)], False),
        # This rank is the only active one and is outside the phase.
        (False, [(1, 1), (0, 0)], False),
    ],
)
def test_mixed_phase_steps_are_skipped(monkeypatch, phase_matched, rank_flags, expected):
    state, upstream_step, _ = _decision_state(monkeypatch, phase_matched=phase_matched)
    sent = []
    monkeypatch.setattr(state_module, "all_reduce", MagicMock(side_effect=_reduced_by(rank_flags, sent)))

    state.step()

    # The rank reports whether it is active and whether its batch matches the phase.
    assert sent == [[1, int(not phase_matched)]]
    _assert_committed(state, upstream_step, expected)


def test_single_replica_ep_group_decides_locally(monkeypatch):
    state, upstream_step, _ = _decision_state(monkeypatch, shared_ep_group=False)
    all_reduce = MagicMock()
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)

    state.step(is_dummy=True)

    all_reduce.assert_not_called()
    _assert_committed(state, upstream_step, False)


@pytest.mark.parametrize("gathered,expected", ((1, True), (0, False)))
def test_allgather_decision_wins_but_the_group_still_meets(monkeypatch, gathered, expected):
    # Participation must not depend on which layers ran on a rank, or the
    # collective would hang. An AllGather layer's gathered source eligibility
    # is more precise than the per-rank phase, so it decides the step.
    state, upstream_step, cpu_group = _decision_state(monkeypatch, gathered=gathered)
    all_reduce = MagicMock(side_effect=_reduced_by([(0, 0), (1, 1)], []))
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)

    state.step(is_dummy=True)

    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["group"] is cpu_group
    _assert_committed(state, upstream_step, expected)


def test_non_sampling_steps_do_not_meet(monkeypatch):
    state, upstream_step, _ = _decision_state(monkeypatch)
    all_reduce = MagicMock()
    monkeypatch.setattr(state_module, "all_reduce", all_reduce)
    state._is_load_sampling_step = False

    state.step(is_dummy=True)

    all_reduce.assert_not_called()
    assert int(state._local_load_collection_mask[0]) == 0
    assert upstream_step.call_args.kwargs["is_dummy"] is True


class TestAscendEplbFreshLoadGate(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "vllm_ascend.distributed.eplb.eplb_state._needs_group_load_decision",
            return_value=False,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _make_state(*, rearrangement_step=1):
        state = object.__new__(AscendEplbState)
        state._hixl_session = None
        state._close_error = None
        state._suspended = False
        state.parallel_config = SimpleNamespace(
            enable_elastic_ep=False,
            eplb_config=SimpleNamespace(log_balancedness_interval=1),
        )
        state.device = torch.device("cpu")
        state.model_states = {}
        state.is_async = False
        state.expert_rearrangement_step = rearrangement_step
        state.expert_rearrangement_step_interval = 2
        state.expert_load_window_step = 0
        state.expert_load_window_size = 2
        state.should_record_tensor = None
        state._has_fresh_recorded_load = False
        return state

    @staticmethod
    def _ep_group():
        return SimpleNamespace(device_group=MagicMock())

    def test_dummy_period_skips_rearrange_but_resets_clock(self):
        state = self._make_state()

        with (
            patch(
                "vllm.distributed.eplb.eplb_state.get_ep_group",
                return_value=self._ep_group(),
            ),
            patch.object(
                state,
                "_has_global_fresh_recorded_load",
                return_value=False,
            ) as sync_fresh_load,
            patch("vllm.distributed.eplb.eplb_state.EplbState.rearrange") as upstream_rearrange,
        ):
            state.step(is_dummy=True)

        self.assertEqual(state.expert_rearrangement_step, 0)
        sync_fresh_load.assert_called_once_with()
        upstream_rearrange.assert_not_called()

    def test_fresh_recorded_load_runs_rearrange_and_is_consumed(self):
        for is_async in (False, True):
            with self.subTest(is_async=is_async):
                state = self._make_state()
                state.is_async = is_async

                with (
                    patch(
                        "vllm.distributed.eplb.eplb_state.get_ep_group",
                        return_value=self._ep_group(),
                    ),
                    patch.object(
                        state,
                        "_has_global_fresh_recorded_load",
                        return_value=True,
                    ) as sync_fresh_load,
                    patch("vllm.distributed.eplb.eplb_state.EplbState.rearrange") as upstream_rearrange,
                ):
                    state.step()

                self.assertEqual(state.expert_rearrangement_step, 0)
                self.assertFalse(state._has_fresh_recorded_load)
                sync_fresh_load.assert_called_once_with()
                upstream_rearrange.assert_called_once_with(
                    is_profile=False,
                    rank_mapping=None,
                )

    def test_remote_fresh_load_enables_all_ranks(self):
        state = self._make_state()
        cpu_group = MagicMock()
        cpu_group.size.return_value = 2
        ep_group = SimpleNamespace(cpu_group=cpu_group)

        def set_remote_fresh_load(flag, **_kwargs):
            flag.fill_(1)

        with (
            patch(
                "vllm_ascend.distributed.eplb.eplb_state.get_ep_group",
                return_value=ep_group,
            ),
            patch(
                "vllm_ascend.distributed.eplb.eplb_state.all_reduce",
                side_effect=set_remote_fresh_load,
            ) as sync_fresh_load,
        ):
            self.assertTrue(state._has_global_fresh_recorded_load())

        sync_fresh_load.assert_called_once()
        self.assertIs(sync_fresh_load.call_args.kwargs["group"], cpu_group)

    def test_profile_and_elastic_rearranges_bypass_gate(self):
        for is_profile, enable_elastic_ep in ((True, False), (False, True)):
            with self.subTest(
                is_profile=is_profile,
                enable_elastic_ep=enable_elastic_ep,
            ):
                state = self._make_state()
                state.parallel_config.enable_elastic_ep = enable_elastic_ep
                state._has_fresh_recorded_load = True

                with (
                    patch.object(
                        state,
                        "_has_global_fresh_recorded_load",
                    ) as sync_fresh_load,
                    patch("vllm.distributed.eplb.eplb_state.EplbState.rearrange") as upstream_rearrange,
                ):
                    state.rearrange(is_profile=is_profile)

                sync_fresh_load.assert_not_called()
                upstream_rearrange.assert_called_once_with(
                    is_profile=is_profile,
                    rank_mapping=None,
                )
                self.assertIs(
                    state._has_fresh_recorded_load,
                    is_profile,
                )
