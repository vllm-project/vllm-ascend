# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.eplb.state import AscendEplbState
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

        with (
            patch("vllm.distributed.eplb.eplb_state.CpuGpuEvent"),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            controller.prepare_load()

        self.assertIsInstance(controller.state, AscendEplbState)

    def test_setup_from_mapping_uses_current_upstream_contract(self):
        controller = self._make_controller()
        model = MagicMock()
        model_config = object()
        mapping = torch.zeros((1, 1), dtype=torch.int32)
        state = MagicMock()

        with (
            patch("vllm_ascend.worker.v2.eplb.is_mixture_of_experts", return_value=True),
            patch.object(AscendEplbState, "from_mapping", return_value=state) as from_mapping,
        ):
            controller.setup_from_mapping(model, model_config, mapping)

        from_mapping.assert_called_once_with(
            model=model,
            model_config=model_config,
            device=controller.device,
            parallel_config=controller.parallel_config,
            expanded_physical_to_logical=mapping,
        )
        self.assertIs(controller.state, state)
        self.assertTrue(controller._has_registered_models)

    def test_setup_from_mapping_accepts_release_upstream_contract(self):
        controller = self._make_controller()
        model = MagicMock()
        model_config = object()
        mapping = torch.zeros((1, 2), dtype=torch.int32)
        state = MagicMock()

        with (
            patch("vllm_ascend.worker.v2.eplb.is_mixture_of_experts", return_value=True),
            patch.object(AscendEplbState, "from_mapping", return_value=state) as from_mapping,
        ):
            controller.setup_from_mapping(model, model_config, mapping, 1)

        from_mapping.assert_called_once_with(
            model=model,
            model_config=model_config,
            device=controller.device,
            parallel_config=controller.parallel_config,
            expanded_physical_to_logical=mapping,
            num_valid_physical_experts=1,
        )
        self.assertIs(controller.state, state)
        self.assertTrue(controller._has_registered_models)

    def test_prepare_forward_combines_window_and_phase_device_gates(self):
        for batch_has_prefill, expected_record in ((False, False), (True, True)):
            with self.subTest(batch_has_prefill=batch_has_prefill):
                controller = self._make_controller(load_collection_phase="prefill")
                state = MagicMock()
                state.should_record_tensor = torch.tensor(True)
                state._has_fresh_recorded_load = False
                state._should_record_current_step.return_value = True
                controller.state = state
                controller.set_batch_phase(batch_has_prefill=batch_has_prefill)

                controller.prepare_forward(object(), 7)

                state.prepare_forward.assert_called_once()
                state._should_record_current_step.assert_called_once_with(log_stats=False)
                self.assertIs(bool(state.should_record_tensor), expected_record)
                self.assertIs(state._has_fresh_recorded_load, expected_record)


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
                "vllm_ascend.distributed.eplb.state.get_ep_group",
                return_value=ep_group,
            ),
            patch(
                "vllm_ascend.distributed.eplb.state.all_reduce",
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
