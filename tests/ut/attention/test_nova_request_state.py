# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.attention.nova_request_state import NovaRequestState
from vllm_ascend.core.nova_scheduler import NovaScheduler


def test_unscheduled_requests_keep_windows_and_reorder():
    state = NovaRequestState(3)
    first = state.prepare(["a", "b"], [0, 0], [2, 3])
    assert state.prepare(["b"], [3], [1]) == [first[1]]
    assert state.prepare(["b", "a"], [4, 2], [1, 1]) == [first[1], first[0]]
    state.release({"b"})
    assert state.prepare(["c"], [0], [2]) == [first[1]]
    assert state.prepare(["a"], [3], [1]) == [first[0]]


def test_missing_and_length_mismatch_require_recompute():
    state = NovaRequestState(2)
    state.prepare(["a", "b"], [0, 0], [3, 4])
    state.release({"a"})
    assert state.missing(["a", "b", "new"], [3, 5, 0]) == {"a", "b"}
    with pytest.raises(RuntimeError, match="recomputed from zero"):
        state.prepare(["a"], [3], [1])
    assert state.prepare(["a"], [0], [3]) == [0]


def test_partial_allocation_does_not_evict_unscheduled_request():
    state = NovaRequestState(2)
    state.prepare(["a", "b"], [0, 0], [1, 1])
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        state.prepare(["c"], [0], [1])
    assert state.missing(["a", "b"], [1, 1]) == set()
    state.release({"a"})
    assert state.prepare(["c"], [0], [1]) == [0]


def test_scheduler_recomputes_only_lost_requests(monkeypatch):
    first = SimpleNamespace(request_id="a", num_computed_tokens=4, status=RequestStatus.RUNNING)
    second = SimpleNamespace(request_id="b", num_computed_tokens=7, status=RequestStatus.RUNNING)
    scheduler = object.__new__(NovaScheduler)
    scheduler.running = [first, second]
    scheduler.nova_executor = SimpleNamespace(collective_rpc=Mock(return_value=[set(), {"b"}]))
    scheduler._preempt_request = Mock()
    monkeypatch.setattr(Scheduler, "schedule", lambda self, throttle: "scheduled")
    assert scheduler.schedule() == "scheduled"
    assert scheduler.running == [first]
    assert scheduler._preempt_request.call_args.args[0] is second


def test_scheduler_does_not_preempt_healthy_requests(monkeypatch):
    request = SimpleNamespace(request_id="a", num_computed_tokens=4, status=RequestStatus.RUNNING)
    scheduler = object.__new__(NovaScheduler)
    scheduler.running = [request]
    scheduler.nova_executor = SimpleNamespace(collective_rpc=Mock(return_value=[set(), set()]))
    scheduler._preempt_request = Mock()
    monkeypatch.setattr(Scheduler, "schedule", lambda self, throttle: "scheduled")
    assert scheduler.schedule() == "scheduled"
    scheduler._preempt_request.assert_not_called()


def test_recovery_resets_progress_and_queues_recompute(monkeypatch):
    request = Request("lost", list(range(6)), SamplingParams(max_tokens=8), None)
    request.append_output_token_ids([6, 7])
    request.num_computed_tokens = 7
    request.status = RequestStatus.RUNNING
    scheduler = object.__new__(NovaScheduler)
    scheduler.running = [request]
    scheduler.nova_executor = SimpleNamespace(collective_rpc=Mock(return_value=[{"lost"}]))
    scheduler.aux_output_connector = None
    scheduler._free_request_blocks = Mock()
    scheduler.encoder_cache_manager = Mock()
    scheduler._pending_encoder_input_ids = {}
    scheduler._inflight_prefills = {request}
    scheduler.waiting = Mock()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    monkeypatch.setattr(Scheduler, "schedule", lambda self, throttle: "scheduled")
    scheduler.schedule()
    scheduler._free_request_blocks.assert_called_once_with(request)
    scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    scheduler.waiting.prepend_request.assert_called_once_with(request)
    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_preemptions == 1
    assert list(request.all_token_ids) == list(range(8))
    assert scheduler.reset_preempted_req_ids == {"lost"}
    state = NovaRequestState(1)
    assert state.prepare([request.request_id], [request.num_computed_tokens], [7]) == [0]
    assert state.missing(["lost"], [7]) == set()
