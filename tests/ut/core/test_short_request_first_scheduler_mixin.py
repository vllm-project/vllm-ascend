#
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
"""Tests for the ShortRequestFirst scheduler mixin glue."""

from types import SimpleNamespace

from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstRequestQueue,
    ShortRequestFirstSchedulerMixin,
)

THRESHOLD = 256


def make_request(
    request_id: str,
    prompt_len: int,
    computed: int = 0,
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_output_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=computed,
        status=status,
        num_output_tokens=num_output_tokens,
    )


class SchedulerStub(ShortRequestFirstSchedulerMixin):
    def __init__(self, waiting=None, skipped_waiting=None, policy=SchedulingPolicy.FCFS):
        self.waiting = waiting
        self.skipped_waiting = skipped_waiting
        self.policy = policy


def _short_request_first_queue(immediate_predicate=None):
    return ShortRequestFirstRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=THRESHOLD,
        long_max_wait_ms=0.0,
        immediate_predicate=immediate_predicate,
    )


def test_select_prefers_skipped_waiting_under_fcfs():
    waiting = _short_request_first_queue()
    waiting.add_request(make_request("s", 10))
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    skipped.add_request(make_request("skipped", 10))

    stub = SchedulerStub(waiting, skipped_waiting=skipped)
    assert stub._select_waiting_queue_for_scheduling() is skipped


def test_select_uses_short_queue_when_skipped_empty():
    waiting = _short_request_first_queue()
    waiting.add_request(make_request("l", THRESHOLD + 100))
    waiting.add_request(make_request("s", 10))
    skipped = create_request_queue(SchedulingPolicy.FCFS)

    stub = SchedulerStub(waiting, skipped_waiting=skipped)
    assert stub._select_waiting_queue_for_scheduling() is waiting._short_queue


def test_select_returns_none_when_everything_empty():
    waiting = _short_request_first_queue()
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    stub = SchedulerStub(waiting, skipped_waiting=skipped)
    assert stub._select_waiting_queue_for_scheduling() is None


def test_owns_queue_distinguishes_managed_subqueues_from_skipped():
    waiting = _short_request_first_queue()
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    assert waiting.owns_queue(waiting._short_queue)
    assert not waiting.owns_queue(skipped)


def test_recovery_markers_route_long_prompt_to_immediate_queue():
    predicate = SchedulerStub()._is_recovery_request
    long_len = THRESHOLD + 100
    cases = {
        "preempted": make_request("preempted", long_len, status=RequestStatus.PREEMPTED),
        "computed": make_request("computed", long_len, computed=1),
        "output": make_request("output", long_len, num_output_tokens=1),
    }
    for name, request in cases.items():
        queue = _short_request_first_queue(immediate_predicate=predicate)
        queue.add_request(request)
        assert queue.num_immediate_requests == 1, name
        assert queue.num_long_requests == 0, name
        assert queue.num_short_requests == 0, name


def test_non_recovery_requests_stay_in_length_based_queues():
    predicate = SchedulerStub()._is_recovery_request

    long_queue = _short_request_first_queue(immediate_predicate=predicate)
    long_queue.add_request(make_request("plain_long", THRESHOLD + 100))
    assert long_queue.num_long_requests == 1
    assert long_queue.num_immediate_requests == 0

    short_queue = _short_request_first_queue(immediate_predicate=predicate)
    short_queue.add_request(make_request("plain_short", 10))
    assert short_queue.num_short_requests == 1
    assert short_queue.num_immediate_requests == 0
