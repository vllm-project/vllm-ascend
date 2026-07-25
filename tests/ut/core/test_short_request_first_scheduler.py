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
"""Unit tests for the ShortRequestFirst waiting queue."""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.short_request_first_scheduler import ShortRequestFirstRequestQueue

THRESHOLD = 256


def make_request(request_id: str, prompt_len: int, **extra) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, num_prompt_tokens=prompt_len, **extra)


def make_queue(
    long_max_wait_ms: float = 0.0,
    threshold: int = THRESHOLD,
    immediate_predicate=None,
) -> ShortRequestFirstRequestQueue:
    return ShortRequestFirstRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=threshold,
        long_max_wait_ms=long_max_wait_ms,
        immediate_predicate=immediate_predicate,
    )


def short_req(rid="short", n=10):
    return make_request(rid, n)


def long_req(rid="long", n=THRESHOLD + 1000):
    return make_request(rid, n)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def test_classify_short_and_long_by_threshold():
    q = make_queue(threshold=256)
    q.add_request(short_req("s", 256))
    q.add_request(long_req("l", 257))
    assert q.num_short_requests == 1
    assert q.num_long_requests == 1
    assert q.num_immediate_requests == 0


def test_immediate_predicate_routes_to_immediate_queue():
    q = make_queue(immediate_predicate=lambda r: r.request_id == "hot")
    q.add_request(make_request("hot", 5))
    assert q.num_immediate_requests == 1
    assert q.num_short_requests == 0


def test_dispatch_priority_immediate_then_short_then_long():
    q = make_queue(immediate_predicate=lambda request: request.request_id == "imm")
    q.add_request(long_req("l"))
    q.add_request(short_req("s"))
    q.add_request(make_request("imm", 5))

    assert q.pop_request().request_id == "imm"
    assert q.pop_request().request_id == "s"
    assert q.pop_request().request_id == "l"


def test_short_runs_before_long_without_aging():
    q = make_queue()
    q.add_request(long_req("l"))
    q.add_request(short_req("s"))
    assert q._select_schedulable_queue() is q._short_queue


def test_aged_long_promotes_over_short_after_wait():
    clock = FakeClock()
    with patch("vllm_ascend.core.short_request_first_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))

        clock.advance(0.15)
        assert q._select_schedulable_queue() is q._long_queue


def test_no_aging_when_long_max_wait_is_zero():
    clock = FakeClock()
    with patch("vllm_ascend.core.short_request_first_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=0.0)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))

        clock.advance(10.0)
        assert q._select_schedulable_queue() is q._short_queue


def test_peek_pins_the_selected_lane_until_pop():
    clock = FakeClock()
    with patch("vllm_ascend.core.short_request_first_scheduler.time", clock):
        q = make_queue(long_max_wait_ms=100.0)
        q.add_request(long_req("l"))
        q.add_request(short_req("s"))

        assert q.peek_request().request_id == "s"
        clock.advance(0.2)

        assert q.pop_request().request_id == "s"
        assert q.pop_request().request_id == "l"


def test_repeated_aged_long_promotions_trigger_warning_and_reset_after_short_dispatch():
    clock = FakeClock()
    with (
        patch("vllm_ascend.core.short_request_first_scheduler.time", clock),
        patch("vllm_ascend.core.short_request_first_scheduler.logger.warning_once") as mock_warning_once,
    ):
        q = make_queue(long_max_wait_ms=100.0)
        q.add_request(long_req("l0"))
        q.add_request(long_req("l1"))
        q.add_request(long_req("l2"))
        q.add_request(short_req("s0"))

        clock.advance(0.2)
        for _ in range(3):
            assert q._select_schedulable_queue() is q._long_queue
            q.pop_request()

        mock_warning_once.assert_called_once()
        assert q._consecutive_aged_long_promotions == 3

        assert q._select_schedulable_queue() is q._short_queue
        q.pop_request()
        assert q._consecutive_aged_long_promotions == 0


def test_stats_log_is_emitted_every_five_seconds():
    clock = FakeClock()
    with (
        patch("vllm_ascend.core.short_request_first_scheduler.time", clock),
        patch("vllm_ascend.core.short_request_first_scheduler.logger.info") as mock_info,
    ):
        q = make_queue()
        q.add_request(short_req("s0"))
        assert mock_info.call_count == 0

        clock.advance(5.1)
        q.add_request(long_req("l0"))
        assert mock_info.call_count == 1

        clock.advance(1.0)
        q.pop_request()
        assert mock_info.call_count == 1

        clock.advance(5.1)
        q.pop_request()
        assert mock_info.call_count == 2
