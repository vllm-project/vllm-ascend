# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_ascend.patch.platform.patch_encoder_cache_free  # noqa: F401
from vllm_ascend.utils import vllm_version_is


class _Request:
    """Minimal hashable stand-in for a scheduler ``Request``."""

    def __init__(self, *, has_encoder_inputs: bool, computed_tokens: int, num_tokens: int):
        self.has_encoder_inputs = has_encoder_inputs
        self.num_computed_tokens = computed_tokens
        self.num_in_flight_tokens = computed_tokens
        self.num_tokens = num_tokens
        self.num_output_placeholders = 0
        self.is_prefill_chunk = True
        self.use_structured_output = False


def _request(*, has_encoder_inputs: bool, computed_tokens: int, num_tokens: int) -> _Request:
    return _Request(
        has_encoder_inputs=has_encoder_inputs,
        computed_tokens=computed_tokens,
        num_tokens=num_tokens,
    )


def _run_update_after_schedule(
    requests: dict[str, _Request],
    num_scheduled_tokens: dict[str, int],
) -> tuple[SimpleNamespace, SimpleNamespace]:
    scheduler = SimpleNamespace(
        requests=requests,
        defer_block_free=False,
        _inflight_prefills=set(),
        _free_encoder_inputs=Mock(),
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        has_structured_output_requests=False,
    )
    Scheduler._update_after_schedule(scheduler, scheduler_output)
    return scheduler, scheduler_output


def test_frees_encoder_inputs_during_schedule():
    encoder_request = _request(has_encoder_inputs=True, computed_tokens=0, num_tokens=4)
    text_request = _request(has_encoder_inputs=False, computed_tokens=0, num_tokens=4)
    scheduler, _ = _run_update_after_schedule(
        {"encoder": encoder_request, "text": text_request},
        {"encoder": 4, "text": 4},
    )

    scheduler._free_encoder_inputs.assert_called_once_with(encoder_request)


def test_frees_after_scheduled_tokens_are_accounted_for():
    encoder_request = _request(has_encoder_inputs=True, computed_tokens=0, num_tokens=4)
    freed_computed_tokens: list[int] = []

    scheduler = SimpleNamespace(
        requests={"encoder": encoder_request},
        defer_block_free=False,
        _inflight_prefills=set(),
        _free_encoder_inputs=Mock(
            side_effect=lambda request: freed_computed_tokens.append(request.num_computed_tokens)
        ),
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"encoder": 2},
        has_structured_output_requests=False,
    )

    Scheduler._update_after_schedule(scheduler, scheduler_output)

    assert freed_computed_tokens == [2]
    assert encoder_request.is_prefill_chunk


def test_frees_each_scheduled_encoder_request_once():
    first = _request(has_encoder_inputs=True, computed_tokens=0, num_tokens=4)
    second = _request(has_encoder_inputs=True, computed_tokens=0, num_tokens=4)
    scheduler, _ = _run_update_after_schedule(
        {"first": first, "second": second},
        {"first": 4, "second": 4},
    )

    assert scheduler._free_encoder_inputs.call_count == 2
    scheduler._free_encoder_inputs.assert_any_call(first)
    scheduler._free_encoder_inputs.assert_any_call(second)


def test_does_not_free_text_only_requests():
    text_request = _request(has_encoder_inputs=False, computed_tokens=0, num_tokens=4)
    scheduler, _ = _run_update_after_schedule({"text": text_request}, {"text": 4})

    scheduler._free_encoder_inputs.assert_not_called()


def test_balance_scheduler_inherits_base_class_patch():
    """Guard the import order: the patch must wrap upstream Scheduler, not the
    BalanceScheduler subclass. Importing it after patch_balance_schedule would
    leave DyntraLB/Recompute schedulers on the unpatched base method."""
    if not vllm_version_is("0.34.0"):
        pytest.skip("upstream vLLM main already frees encoder inputs at schedule time")

    from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

    base_scheduler = BalanceScheduler.__mro__[1]
    assert "_update_after_schedule" not in BalanceScheduler.__dict__
    assert getattr(base_scheduler._update_after_schedule, "_vllm_ascend_encoder_free_patch", False)
