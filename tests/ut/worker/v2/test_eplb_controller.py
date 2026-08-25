# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces import SupportsMultiModal

from vllm_ascend.worker.v2.eplb import AscendEPLBController, _unwrap_moe


class TestAscendEPLBController(unittest.TestCase):
    @staticmethod
    def _make_controller(*, enable_eplb=True, log_balancedness=False):
        parallel_config = SimpleNamespace(
            enable_eplb=enable_eplb,
            eplb_config=SimpleNamespace(
                log_balancedness=log_balancedness,
            ),
        )
        controller = AscendEPLBController(
            parallel_config,
            torch.device("cpu"),
        )
        controller._has_registered_models = True
        return controller

    def test_prepare_load_resets_state_when_eplb_is_disabled(self):
        controller = self._make_controller(enable_eplb=False)
        controller.state = MagicMock()

        with patch("vllm_ascend.worker.v2.eplb.AscendEplbState") as ascend_state:
            controller.prepare_load()

        self.assertIsNone(controller.state)
        self.assertFalse(controller._has_registered_models)
        ascend_state.assert_not_called()

    def test_prepare_load_constructs_ascend_state(self):
        controller = self._make_controller()
        state = MagicMock()

        with patch(
            "vllm_ascend.worker.v2.eplb.AscendEplbState",
            return_value=state,
        ) as ascend_state:
            controller.prepare_load()

        self.assertIs(controller.state, state)
        self.assertFalse(controller._has_registered_models)
        ascend_state.assert_called_once_with(
            controller.parallel_config,
            controller.device,
        )

    def test_set_batch_phase_updates_match(self):
        controller = self._make_controller()
        controller.load_collection_phase = "prefill"

        controller.set_batch_phase(batch_has_prefill=True)
        self.assertTrue(controller._load_collection_phase_matched)

        controller.set_batch_phase(batch_has_prefill=False)
        self.assertFalse(controller._load_collection_phase_matched)

    def test_step_early_return_conditions(self):
        for condition in (
            "disabled",
            "suppressed",
            "missing_state",
            "unregistered",
        ):
            with self.subTest(condition=condition):
                controller = self._make_controller()
                state = MagicMock()
                controller.state = state

                if condition == "disabled":
                    controller.parallel_config.enable_eplb = False
                elif condition == "suppressed":
                    controller.suppressed = True
                elif condition == "missing_state":
                    controller.state = None
                else:
                    controller._has_registered_models = False

                controller.step()

                state._should_record_current_step.assert_not_called()
                state.step.assert_not_called()

    def test_dummy_and_profile_steps_skip_window_check(self):
        for is_dummy, is_profile in ((True, False), (False, True)):
            with self.subTest(is_dummy=is_dummy, is_profile=is_profile):
                controller = self._make_controller(log_balancedness=True)
                state = MagicMock()
                controller.state = state

                controller.step(is_dummy=is_dummy, is_profile=is_profile)

                state._should_record_current_step.assert_not_called()
                state.step.assert_called_once_with(
                    is_dummy,
                    is_profile,
                    log_stats=True,
                )

    def test_nonmatching_phase_is_forwarded_as_dummy(self):
        controller = self._make_controller(log_balancedness=True)
        controller.load_collection_phase = "prefill"
        controller.set_batch_phase(batch_has_prefill=False)
        state = MagicMock()
        controller.state = state

        controller.step()

        state._should_record_current_step.assert_not_called()
        state.step.assert_called_once_with(True, False, log_stats=True)

    def test_open_window_preserves_recorded_load(self):
        controller = self._make_controller(log_balancedness=True)
        expert_load_pass = torch.ones(2, dtype=torch.int32)
        state = MagicMock()
        state._should_record_current_step.return_value = True
        state.model_states = {"model": SimpleNamespace(expert_load_pass=expert_load_pass)}
        controller.state = state

        controller.step()

        torch.testing.assert_close(
            expert_load_pass,
            torch.ones_like(expert_load_pass),
        )
        state.step.assert_called_once_with(False, False, log_stats=True)

    def test_closed_window_clears_every_registered_model(self):
        controller = self._make_controller()
        first_load = torch.ones(2, dtype=torch.int32)
        second_load = torch.ones(3, dtype=torch.int32)
        state = MagicMock()
        state._should_record_current_step.return_value = False
        state.model_states = {
            "first": SimpleNamespace(expert_load_pass=first_load),
            "second": SimpleNamespace(expert_load_pass=second_load),
        }
        controller.state = state

        controller.step()

        torch.testing.assert_close(first_load, torch.zeros_like(first_load))
        torch.testing.assert_close(second_load, torch.zeros_like(second_load))

    def test_setup_from_mapping_constructs_state_and_registers_model(self):
        controller = self._make_controller()
        model = nn.Linear(2, 2)
        model_config = SimpleNamespace()
        mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        state = MagicMock()

        with (
            patch(
                "vllm_ascend.worker.v2.eplb._unwrap_moe",
                return_value=model,
            ) as unwrap_moe,
            patch(
                "vllm_ascend.worker.v2.eplb.is_mixture_of_experts",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.v2.eplb.AscendEplbState.from_mapping",
                return_value=state,
            ) as from_mapping,
        ):
            controller.setup_from_mapping(
                model=model,
                model_config=model_config,
                expanded_physical_to_logical=mapping,
                old_num_physical_experts=2,
            )

        unwrap_moe.assert_called_once_with(model)
        from_mapping.assert_called_once_with(
            model=model,
            model_config=model_config,
            device=controller.device,
            parallel_config=controller.parallel_config,
            expanded_physical_to_logical=mapping,
            num_valid_physical_experts=2,
        )
        self.assertIs(controller.state, state)
        self.assertTrue(controller._has_registered_models)

    def test_setup_from_mapping_rejects_non_moe_model(self):
        controller = self._make_controller()
        model = nn.Linear(2, 2)

        with (
            patch(
                "vllm_ascend.worker.v2.eplb._unwrap_moe",
                return_value=model,
            ),
            patch(
                "vllm_ascend.worker.v2.eplb.is_mixture_of_experts",
                return_value=False,
            ),
            self.assertRaises(AssertionError),
        ):
            controller.setup_from_mapping(
                model=model,
                model_config=SimpleNamespace(),
                expanded_physical_to_logical=torch.tensor([0]),
                old_num_physical_experts=1,
            )


class TestUnwrapMoe(unittest.TestCase):
    def test_unwraps_multimodal_non_moe_model(self):
        model = MagicMock(spec=SupportsMultiModal)
        language_model = nn.Linear(2, 2)
        model.get_language_model.return_value = language_model

        with patch(
            "vllm_ascend.worker.v2.eplb.is_mixture_of_experts",
            return_value=False,
        ):
            result = _unwrap_moe(model)

        self.assertIs(result, language_model)
        model.get_language_model.assert_called_once_with()

    def test_keeps_top_level_moe_model(self):
        model = MagicMock(spec=SupportsMultiModal)

        with patch(
            "vllm_ascend.worker.v2.eplb.is_mixture_of_experts",
            return_value=True,
        ):
            result = _unwrap_moe(model)

        self.assertIs(result, model)
        model.get_language_model.assert_not_called()

    def test_keeps_non_multimodal_model(self):
        model = nn.Linear(2, 2)

        with patch(
            "vllm_ascend.worker.v2.eplb.is_mixture_of_experts",
            return_value=False,
        ):
            result = _unwrap_moe(model)

        self.assertIs(result, model)


if __name__ == "__main__":
    unittest.main()
