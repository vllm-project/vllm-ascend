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
