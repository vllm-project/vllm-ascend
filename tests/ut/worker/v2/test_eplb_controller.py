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

    def test_prepare_forward_records_matching_phase(self):
        controller = self._make_controller(log_balancedness=True)
        controller.load_collection_phase = "prefill"
        controller.set_batch_phase(batch_has_prefill=True)
        state = MagicMock()
        state.should_record_tensor = torch.zeros((), dtype=torch.bool)
        state._should_record_current_step.return_value = True
        state._has_fresh_recorded_load = False
        controller.state = state
        model_config = SimpleNamespace()
        ubatch_slices = [slice(0, 4)]

        controller.prepare_forward(model_config, 4, ubatch_slices)

        state.prepare_forward.assert_called_once_with(
            model_config,
            4,
            ubatch_slices,
        )
        state._should_record_current_step.assert_called_once_with(
            log_stats=True,
        )
        self.assertTrue(state.should_record_tensor.item())
        self.assertTrue(state._has_fresh_recorded_load)

    def test_prepare_forward_disables_nonmatching_phase(self):
        controller = self._make_controller(log_balancedness=True)
        controller.load_collection_phase = "prefill"
        controller.set_batch_phase(batch_has_prefill=False)
        state = MagicMock()
        state.should_record_tensor = torch.ones((), dtype=torch.bool)
        state._should_record_current_step.return_value = True
        state._has_fresh_recorded_load = False
        controller.state = state

        controller.prepare_forward(SimpleNamespace(), 4)

        state._should_record_current_step.assert_called_once_with(
            log_stats=True,
        )
        self.assertFalse(state.should_record_tensor.item())
        self.assertFalse(state._has_fresh_recorded_load)

    def test_prepare_forward_disables_closed_window(self):
        controller = self._make_controller()
        state = MagicMock()
        state.should_record_tensor = torch.ones((), dtype=torch.bool)
        state._should_record_current_step.return_value = False
        state._has_fresh_recorded_load = False
        controller.state = state

        controller.prepare_forward(SimpleNamespace(), 4)

        state._should_record_current_step.assert_called_once_with(
            log_stats=False,
        )
        self.assertFalse(state.should_record_tensor.item())
        self.assertFalse(state._has_fresh_recorded_load)

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
