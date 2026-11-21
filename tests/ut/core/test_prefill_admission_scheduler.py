# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from vllm.v1.request import RequestStatus

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import PrefillAdmissionConfig, SchedulerConfig
from vllm_ascend.core.prefill_admission_scheduler import PrefillAdmissionController, _PrefillAdmissionSchedulerMixin


def _decode_request(request_id: str, token_demand: int = 1):
    return SimpleNamespace(
        request_id=request_id,
        is_prefill_chunk=False,
        num_tokens_with_spec=token_demand,
        num_output_placeholders=0,
        num_computed_tokens=0,
        next_decode_eligible_step=0,
    )


def _prefill_request(request_id: str):
    return SimpleNamespace(request_id=request_id)


class TestPrefillAdmissionController(TestBase):
    def test_nested_config_uses_typed_defaults(self):
        config = SchedulerConfig.from_additional_config(
            {"scheduler_config": {"prefill_admission_config": {"enabled": True, "prefill_interval": 7}}}
        )
        self.assertIsInstance(config.prefill_admission_config, PrefillAdmissionConfig)
        self.assertEqual(config.prefill_admission_config.prefill_interval, 7)
        self.assertFalse(SchedulerConfig.from_additional_config({}).prefill_admission_config.enabled)

    def test_invalid_limits_and_unknown_fields_are_rejected(self):
        for values in (
            {"prefill_interval": 0},
            {"decode_low_watermark": -1},
            {"max_prefill_wait_ms": float("nan")},
            {"max_prefill_wait_ms": float("inf")},
            {"prefill_tokens_per_pp_bubble": 0},
            {"unknown_field": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PrefillAdmissionConfig(**values)

    def test_pending_prefills_include_current_deferred_queue(self):
        waiting = SimpleNamespace(
            request_id="waiting", status=RequestStatus.WAITING, num_computed_tokens=0, num_tokens=128
        )
        deferred = SimpleNamespace(
            request_id="deferred", status=RequestStatus.PREEMPTED, num_computed_tokens=0, num_tokens=128
        )
        scheduler = SimpleNamespace(
            running=[],
            waiting=[waiting],
            deferred_waiting=[deferred],
            _waiting_request_needs_prefill=_PrefillAdmissionSchedulerMixin._waiting_request_needs_prefill,
        )
        self.assertEqual(
            [request.request_id for request in _PrefillAdmissionSchedulerMixin._pending_prefills(scheduler)],
            ["waiting", "deferred"],
        )

    def _make_controller(self, now, **overrides):
        user_config = {
            "enabled": True,
            "prefill_interval": 4,
            "decode_low_watermark": 2,
            "max_prefill_wait_ms": 1000,
            "prefill_tokens_per_pp_bubble": 512,
        }
        user_config.update(overrides)
        return PrefillAdmissionController(
            PrefillAdmissionConfig(**user_config),
            pipeline_parallel_size=2,
            clock=lambda: now[0],
        )

    def test_throttles_prefill_while_decode_load_is_sufficient(self):
        now = [0.0]
        controller = self._make_controller(now)

        decision = controller.decide(
            [_decode_request("d0"), _decode_request("d1")],
            [_prefill_request("p0")],
            scheduler_step=1,
            max_token_budget=4096,
        )

        self.assertTrue(decision.throttle_prefills)
        self.assertIsNone(decision.token_budget)
        self.assertEqual(decision.reason, "decode_priority")

    def test_uses_small_token_budget_for_pp_bubble(self):
        now = [0.0]
        controller = self._make_controller(now)

        decision = controller.decide(
            [_decode_request("d0")],
            [_prefill_request("p0")],
            scheduler_step=1,
            max_token_budget=4096,
        )

        self.assertFalse(decision.throttle_prefills)
        self.assertEqual(decision.reason, "pp_bubble")
        self.assertEqual(decision.token_budget, 513)

    def test_periodic_release_does_not_create_a_scheduler_phase(self):
        now = [0.0]
        controller = self._make_controller(now, max_prefill_wait_ms=10000)
        running = [_decode_request("d0"), _decode_request("d1")]
        pending = [_prefill_request("p0")]

        decisions = [
            controller.decide(
                running,
                pending,
                scheduler_step=step,
                max_token_budget=4096,
            )
            for step in range(1, 5)
        ]

        self.assertTrue(all(decision.throttle_prefills for decision in decisions[:3]))
        self.assertFalse(decisions[3].throttle_prefills)
        self.assertEqual(decisions[3].reason, "periodic")
        self.assertEqual(decisions[3].token_budget, 514)

    def test_max_wait_releases_prefill_and_observe_resets_aging(self):
        now = [0.0]
        controller = self._make_controller(now, prefill_interval=100)
        running = [_decode_request("d0"), _decode_request("d1")]
        pending = [_prefill_request("p0")]

        first = controller.decide(running, pending, scheduler_step=1, max_token_budget=4096)
        self.assertTrue(first.throttle_prefills)

        now[0] = 1.1
        released = controller.decide(running, pending, scheduler_step=2, max_token_budget=4096)
        self.assertFalse(released.throttle_prefills)
        self.assertEqual(released.reason, "max_wait")

        controller.observe(released, SimpleNamespace(num_scheduled_tokens={"p0": 512}))
        next_decision = controller.decide(running, pending, scheduler_step=3, max_token_budget=4096)
        self.assertTrue(next_decision.throttle_prefills)

    def test_no_prefill_leaves_upstream_budget_unchanged(self):
        now = [0.0]
        controller = self._make_controller(now)

        decision = controller.decide(
            [_decode_request("d0")],
            [],
            scheduler_step=1,
            max_token_budget=4096,
        )

        self.assertFalse(decision.throttle_prefills)
        self.assertIsNone(decision.token_budget)
        self.assertEqual(decision.reason, "no_prefill")
