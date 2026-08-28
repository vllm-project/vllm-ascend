# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.eplb_state import AscendEplbState
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
    def _make_controller(load_collection_phase="all", log_balancedness=False):
        parallel_config = SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(log_balancedness=log_balancedness),
        )
        controller = AscendEPLBController(
            parallel_config,
            torch.device("cpu"),
            load_collection_phase=load_collection_phase,
        )
        controller._has_registered_models = True
        return controller

    def test_prepare_load_constructs_ascend_state(self):
        controller = self._make_controller()

        with patch("vllm.distributed.eplb.eplb_state.CpuGpuEvent"):
            controller.prepare_load()

        self.assertIsInstance(controller.state, AscendEplbState)

    def test_rank_local_phase_filter_preserves_global_stats_schedule(self):
        for batch_has_prefill, expected_dummy in ((True, False), (False, True)):
            with self.subTest(batch_has_prefill=batch_has_prefill):
                controller = self._make_controller(
                    load_collection_phase="prefill",
                    log_balancedness=True,
                )
                state = MagicMock()
                state._should_record_current_step.return_value = True
                controller.state = state
                controller.set_batch_phase(batch_has_prefill=batch_has_prefill)

                controller.step()

                state.step.assert_called_once_with(expected_dummy, False, log_stats=True)

    def test_closed_upstream_window_discards_recorded_load(self):
        controller = self._make_controller()
        expert_load_pass = torch.ones(2, dtype=torch.int32)
        state = MagicMock()
        state._should_record_current_step.return_value = False
        state.model_states = {"model": SimpleNamespace(expert_load_pass=expert_load_pass)}
        controller.state = state

        controller.step()

        torch.testing.assert_close(
            expert_load_pass,
            torch.zeros_like(expert_load_pass),
        )
        state.step.assert_called_once_with(False, False, log_stats=False)

    def test_suppressed_controller_does_not_touch_state(self):
        controller = self._make_controller()
        controller.suppressed = True
        state = MagicMock()
        controller.state = state

        controller.step()

        state._should_record_current_step.assert_not_called()
        state.step.assert_not_called()


class TestAscendEplbFreshLoadGate(unittest.TestCase):
    @staticmethod
    def _make_state(*, rearrangement_step=1):
        state = object.__new__(AscendEplbState)
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
                "vllm_ascend.distributed.eplb_state.get_ep_group",
                return_value=ep_group,
            ),
            patch(
                "vllm_ascend.distributed.eplb_state.all_reduce",
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
