# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

import vllm_ascend.patch.platform.patch_pp_mtp  # noqa: F401
from vllm_ascend.spec_decode.dynamic.policy import AdaptiveDraftKController


def test_compact_and_expanded_configs_create_equivalent_controllers(monkeypatch):
    import random

    from vllm.v1.core.sched import scheduler as scheduler_module

    from vllm_ascend.core.dynamic_spec_scheduler import install_scheduler_policy
    from vllm_ascend.dynamic_spec_config import resolve_method_params
    from vllm_ascend.worker.v2.spec_decode.physical_k import configured_capture_k, v2_varlen_physical_k_enabled

    class FakeScheduler:
        def __init__(self, vllm_config):
            pass

        def _update_after_schedule(self, output):
            pass

    monkeypatch.setattr(scheduler_module, "Scheduler", FakeScheduler)
    install_scheduler_policy()
    compact = {"method": "dspark", "policy": "hardware_aware", "physical_k": {"min_k": 3, "capture_k": [3, 5]}}
    expanded = {"method": "dspark", "policy": "hardware_aware", "method_params": resolve_method_params(compact)}
    controllers = []
    for dynamic in (compact, expanded):
        config = SimpleNamespace(
            additional_config={"dynamic_spec_config": dynamic},
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            use_v2_model_runner=True,
        )
        controller = FakeScheduler(config)._ascend_physical_k_controller
        assert controller is not None
        controllers.append(controller)
        assert v2_varlen_physical_k_enabled(config)
        assert configured_capture_k(config, 5) == (3, 5)
    rng = random.Random(20260908)
    for _ in range(1000):
        batch = rng.choice([1, 4, 8, 16])
        k = controllers[0].cap(5)
        samples = [list(range(rng.randrange(k + 1) + 1)) for _ in range(batch)]
        for controller in controllers:
            controller.observe([k] * batch, samples)
        assert controllers[0].__dict__ == controllers[1].__dict__


@pytest.mark.parametrize("configured,expected", [(5, 4), (3, 3), (0, 0)])
def test_next_physical_k_is_selected_before_async_placeholders(configured, expected):
    scheduler, request = _scheduler()
    controller = AdaptiveDraftKController(max_k=5, min_k=4, slack=0)
    controller.update([1] * 16)
    scheduler._ascend_physical_k_controller = controller
    output = _output(configured)

    scheduler._update_after_schedule(output)

    assert output.num_spec_tokens_to_schedule == expected
    assert len(request.spec_token_ids) == expected
    assert len(scheduler._spec_token_placeholders) == expected
    # Current-step verification and bonus accounting must not be retroactively
    # shortened when choosing the NEXT step's physical draft width.
    assert len(output.scheduled_spec_decode_tokens["r"]) == 5
    assert request.num_output_placeholders == 6
    assert request.next_decode_eligible_step == 8


def test_upstream_placeholder_width_unchanged_without_controller():
    scheduler, request = _scheduler()
    output = _output(5)
    scheduler._update_after_schedule(output)
    assert len(request.spec_token_ids) == 5
    assert output.num_spec_tokens_to_schedule == 5


def _scheduler():
    scheduler = AsyncScheduler.__new__(AsyncScheduler)
    request = SimpleNamespace(
        num_computed_tokens=10,
        num_in_flight_tokens=0,
        num_tokens=10,
        num_output_placeholders=0,
        use_structured_output=False,
    )
    scheduler.requests = {"r": request}
    scheduler.defer_block_free = False
    scheduler.enable_return_routed_experts = False
    scheduler._inflight_prefills = SimpleNamespace(discard=lambda request: None)
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = True
    scheduler.current_step = 7
    scheduler.pp_size = 1
    return scheduler, request


def _output(configured):
    return SimpleNamespace(
        num_spec_tokens_to_schedule=configured,
        num_scheduled_tokens={"r": 6},
        scheduled_spec_decode_tokens={"r": [0] * 5},
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
    )
