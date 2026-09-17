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


class TestLegacyFlashLBController(unittest.TestCase):
    def make_controller(self):
        controller = AscendEPLBController(
            SimpleNamespace(enable_eplb=False, tensor_parallel_size=8),
            torch.device("cpu"),
            legacy_config=SimpleNamespace(eplb_policy_type=3),
        )
        controller.legacy_updator = MagicMock()
        controller.legacy_updator.update_expert_weight_flag.return_value = False
        controller.legacy_load_enabled = torch.zeros((), dtype=torch.int32)
        controller.legacy_counter_enabled = torch.zeros((), dtype=torch.int32)
        return controller

    def test_real_and_idle_steps_advance_same_planner_without_dummy_load(self):
        for is_dummy in (False, True):
            with self.subTest(is_dummy=is_dummy):
                controller = self.make_controller()
                controller._legacy_is_dummy = is_dummy
                controller.prepare_forward(SimpleNamespace(), 4)
                self.assertEqual(controller.legacy_load_enabled.item(), int(not is_dummy))
                self.assertEqual(controller.legacy_counter_enabled.item(), 1)
                controller.legacy_updator.forward_before.assert_called_once()
                controller.step(is_dummy=is_dummy)
                controller.step(is_dummy=is_dummy)
                controller.legacy_updator.forward_end.assert_called_once()
                self.assertEqual(controller.legacy_load_enabled.item(), 0)
                self.assertEqual(controller.legacy_counter_enabled.item(), 0)

    def test_profile_and_capture_suppression_excludes_collection_and_update(self):
        controller = self.make_controller()
        controller.legacy_load_enabled.fill_(1)
        controller.legacy_counter_enabled.fill_(1)
        with controller.suppress_legacy():
            with controller.suppress_legacy():
                controller.prepare_forward(SimpleNamespace(), 4)
                controller.step(is_dummy=True)
            self.assertTrue(controller.suppressed)
            self.assertEqual(controller.legacy_load_enabled.item(), 0)
            self.assertEqual(controller.legacy_counter_enabled.item(), 0)
        self.assertFalse(controller.suppressed)
        controller.legacy_updator.forward_before.assert_not_called()
        controller.legacy_updator.forward_end.assert_not_called()

    def test_weight_replacement_waits_for_forward_before_update(self):
        controller = self.make_controller()
        controller.legacy_updator.update_expert_weight_flag.return_value = True
        calls = []
        controller.legacy_updator.forward_end.side_effect = lambda: calls.append("update")
        controller.prepare_forward(SimpleNamespace(), 4)
        with patch("torch.npu.current_stream") as current_stream:
            current_stream.return_value.synchronize.side_effect = lambda: calls.append("synchronize")
            controller.step()
        self.assertEqual(calls, ["synchronize", "update"])

    def test_registration_uses_real_policy3_and_shared_graph_gates(self):
        controller = self.make_controller()
        layers = [nn.Linear(2, 2), nn.Linear(2, 2)]
        with (
            patch("vllm_ascend.worker.v2.eplb.Manager"),
            patch("vllm_ascend.worker.v2.eplb.VllmEplbAdaptor") as adaptor,
            patch("vllm_ascend.worker.v2.eplb.D2DExpertWeightLoader"),
            patch("vllm_ascend.worker.v2.eplb.EplbProcess") as process,
            patch("vllm_ascend.worker.v2.eplb.EplbUpdator") as updator,
        ):
            adaptor.return_value.moe_layers = layers
            added = controller.maybe_register_model(nn.Linear(2, 2), SimpleNamespace(), False)
        self.assertFalse(added)
        self.assertEqual(process.call_args.kwargs["policy_type"], 3)
        self.assertEqual(process.call_args.kwargs["tp_size"], 8)
        self.assertIsNone(controller.state)
        self.assertIs(layers[0].eplb_load_enabled, layers[1].eplb_load_enabled)
        self.assertIs(layers[0].eplb_counter_enabled, layers[1].eplb_counter_enabled)
        updator.return_value.warm_up_eplb.assert_called_once()


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
