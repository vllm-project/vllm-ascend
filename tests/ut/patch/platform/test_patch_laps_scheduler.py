from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_ascend.core.laps_scheduler import LAPSRequestQueue


def make_request(request_id: str, prompt_len: int):
    return SimpleNamespace(request_id=request_id, num_prompt_tokens=prompt_len)


def test_short_requests_run_before_long_requests():
    queue = LAPSRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=16,
        wait_window_ms=0,
        wait_max_batch=4,
    )
    long_request = make_request("long", 64)
    short_request = make_request("short", 8)

    queue.add_request(long_request)
    queue.add_request(short_request)

    assert queue
    assert queue.peek_request() is short_request
    assert queue.pop_request() is short_request
    assert queue.pop_request() is long_request


def test_wait_window_holds_isolated_short_request():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=4,
        )
        short_request = make_request("short", 8)
        queue.add_request(short_request)

        assert len(queue) == 1
        assert queue
        assert queue.select_waiting_queue_for_scheduling() is None
        current_time["value"] = 0.009
        assert len(queue) == 1
        assert queue
        assert queue.select_waiting_queue_for_scheduling() is None
        current_time["value"] = 0.011
        assert queue
        assert queue.peek_request() is short_request


def test_long_request_can_run_while_short_batch_is_waiting():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=4,
        )
        short_request = make_request("short", 8)
        long_request = make_request("long", 64)
        queue.add_request(short_request)
        queue.add_request(long_request)

        assert queue
        assert queue.peek_request() is long_request
        assert queue.pop_request() is long_request

        assert not queue
        current_time["value"] = 0.011
        assert queue
        assert queue.pop_request() is short_request


def test_short_batch_dispatches_early_when_target_batch_size_is_reached():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=2,
        )
        short_request_1 = make_request("short-1", 8)
        short_request_2 = make_request("short-2", 12)
        queue.add_request(short_request_1)
        assert not queue

        queue.add_request(short_request_2)
        assert queue
        assert queue.pop_request() is short_request_1
        assert queue.pop_request() is short_request_2


def test_immediate_requests_bypass_short_wait_window():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=4,
            immediate_predicate=lambda req: getattr(req, "immediate", False),
        )
        short_request = SimpleNamespace(
            request_id="short",
            num_prompt_tokens=8,
            immediate=False,
        )
        resumed_request = SimpleNamespace(
            request_id="resumed",
            num_prompt_tokens=8,
            immediate=True,
        )
        queue.add_request(short_request)
        queue.add_request(resumed_request)

        assert queue
        assert queue.peek_request() is resumed_request
        assert queue.pop_request() is resumed_request
        assert queue
        assert queue.select_waiting_queue_for_scheduling() is None

        current_time["value"] = 0.011
        assert queue
        assert queue.pop_request() is short_request


def test_select_waiting_queue_for_scheduling_uses_internal_subqueues():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=4,
        )
        short_request = make_request("short", 8)
        long_request = make_request("long", 64)
        queue.add_request(short_request)
        queue.add_request(long_request)

        assert queue.select_waiting_queue_for_scheduling() is not None
        assert queue.select_waiting_queue_for_scheduling().peek_request() is long_request

        queue.pop_request()
        assert queue
        assert queue.select_waiting_queue_for_scheduling() is None


def test_prepend_request_can_force_immediate_queue():
    current_time = {"value": 0.0}

    with patch(
        "vllm_ascend.core.laps_scheduler.time.monotonic",
        side_effect=lambda: current_time["value"],
    ):
        queue = LAPSRequestQueue(
            policy=SchedulingPolicy.FCFS,
            threshold=16,
            wait_window_ms=10.0,
            wait_max_batch=4,
            immediate_predicate=lambda req: getattr(req, "immediate", False),
        )
        short_request = SimpleNamespace(
            request_id="short",
            num_prompt_tokens=8,
            immediate=False,
        )
        preempted_request = SimpleNamespace(
            request_id="preempted",
            num_prompt_tokens=8,
            immediate=False,
        )

        queue.add_request(short_request)
        queue.prepend_request(preempted_request, force_immediate=True)

        assert queue.peek_request() is preempted_request
        assert queue.pop_request() is preempted_request
        assert queue.select_waiting_queue_for_scheduling() is None


def test_remove_request_uses_request_id_instead_of_identity():
    queue = LAPSRequestQueue(
        policy=SchedulingPolicy.FCFS,
        threshold=16,
        wait_window_ms=0,
        wait_max_batch=4,
    )
    request = make_request("shared-id", 8)
    same_id_different_object = make_request("shared-id", 8)

    queue.add_request(request)

    assert same_id_different_object in queue
    queue.remove_request(same_id_different_object)
    assert len(queue) == 0
