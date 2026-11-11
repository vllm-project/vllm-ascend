# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.ut.core.test_scheduler_pause_drain import BLOCK_SIZE, MAX_DRAIN_STEPS, make_scheduler, start_async_load, step
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

BUDGET_SCHEDULERS = (
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    BalanceScheduler,
    RecomputeScheduler,
    ProfilingChunkScheduler,
)


@pytest.fixture(autouse=True)
def reset_ascend_config() -> Iterator[None]:
    clear_ascend_config()
    yield
    clear_ascend_config()


@pytest.fixture(params=BUDGET_SCHEDULERS, ids=lambda cls: cls.__name__)
def scheduler_cls(request) -> type[Scheduler]:
    return request.param


@pytest.fixture
def scheduler(scheduler_cls: type[Scheduler]) -> Scheduler:
    return make_scheduler(scheduler_cls)


def configure_budgets(scheduler, monkeypatch, *, token_budget, input_budget, draft_slots):
    monkeypatch.setattr(scheduler, "max_num_scheduled_tokens", token_budget)
    monkeypatch.setattr(scheduler.scheduler_config, "max_num_batched_tokens", input_budget)
    # Isolate the scheduler's extra drafting-slot contract from draft-model setup.
    spec = SimpleNamespace(max_num_new_slots_for_drafting=draft_slots) if draft_slots else None
    monkeypatch.setattr(scheduler.vllm_config, "speculative_config", spec)


@pytest.mark.parametrize(
    ("token_budget", "input_budget", "draft_slots", "expected"),
    [
        (8, 8, 0, [6, 2]),
        (8, 8, 2, [6, 0]),
        (8, 7, 2, [5, 0]),
        (3, 12, 2, [3, 0]),
        (8, 2, 2, [0, 0]),
        (8, 3, 2, [1, 0]),
    ],
)
def test_waiting_requests_respect_both_budgets(
    scheduler, monkeypatch, token_budget, input_budget, draft_slots, expected
):
    requests = [create_request(i, num_tokens=6, block_size=BLOCK_SIZE) for i in (1, 2)]
    for request in requests:
        scheduler.add_request(request)
    configure_budgets(
        scheduler, monkeypatch, token_budget=token_budget, input_budget=input_budget, draft_slots=draft_slots
    )

    output = scheduler.schedule()

    assert [output.num_scheduled_tokens.get(request.request_id, 0) for request in requests] == expected
    for request, num_tokens in zip(requests, expected):
        assert (request in scheduler.running) == (num_tokens > 0)


@pytest.mark.parametrize(
    ("token_budget", "input_budget", "draft_slots", "expected"),
    [(8, 8, 0, [1, 1]), (8, 3, 2, [1, 0]), (8, 2, 2, [0, 0]), (1, 8, 2, [1, 0])],
)
def test_running_requests_respect_both_budgets(
    scheduler, monkeypatch, token_budget, input_budget, draft_slots, expected
):
    requests = [create_request(i, num_tokens=4, block_size=BLOCK_SIZE) for i in (1, 2)]
    for request in requests:
        scheduler.add_request(request)
    step(scheduler)
    configure_budgets(
        scheduler, monkeypatch, token_budget=token_budget, input_budget=input_budget, draft_slots=draft_slots
    )

    output = scheduler.schedule()

    assert [output.num_scheduled_tokens.get(request.request_id, 0) for request in requests] == expected


def test_running_prefill_is_limited_by_remaining_input_budget(scheduler, monkeypatch):
    request = create_request(1, num_tokens=6, block_size=BLOCK_SIZE)
    scheduler.add_request(request)
    configure_budgets(scheduler, monkeypatch, token_budget=2, input_budget=8, draft_slots=0)
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 2}
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=5, draft_slots=2)

    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 3}


def test_running_and_waiting_requests_share_input_budget(scheduler, monkeypatch):
    running = create_request(1, num_tokens=4, block_size=BLOCK_SIZE)
    scheduler.add_request(running)
    step(scheduler)
    waiting = create_request(2, num_tokens=6, block_size=BLOCK_SIZE)
    scheduler.add_request(waiting)
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=6, draft_slots=2)

    assert scheduler.schedule().num_scheduled_tokens == {running.request_id: 1, waiting.request_id: 1}


@pytest.mark.parametrize(("input_budget", "expected_tokens"), [(7, 0), (8, 6)])
def test_unchunked_prefill_requires_enough_input_budget(scheduler, monkeypatch, input_budget, expected_tokens):
    request = create_request(1, num_tokens=6, block_size=BLOCK_SIZE)
    scheduler.add_request(request)
    scheduler.scheduler_config.enable_chunked_prefill = False
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=input_budget, draft_slots=2)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens.get(request.request_id, 0) == expected_tokens
    assert (request in scheduler.waiting) == (expected_tokens == 0)


@pytest.mark.parametrize("scheduler_cls", [DyntraLBScheduler, AsyncDyntraLBScheduler], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(("input_budget", "expected_tokens"), [(5, 0), (6, 4)])
def test_dyntra_padding_respects_input_budget(scheduler, monkeypatch, input_budget, expected_tokens):
    request = create_request(1, num_tokens=1, block_size=BLOCK_SIZE)
    scheduler.add_request(request)
    scheduler.num_spec_tokens = 3
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=input_budget, draft_slots=2)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens.get(request.request_id, 0) == expected_tokens
    assert output.scheduled_spec_decode_tokens.get(request.request_id, []) == ([-1] * 3 if expected_tokens else [])


@pytest.mark.parametrize(
    "scheduler_cls", [DyntraLBScheduler, AsyncDyntraLBScheduler, BalanceScheduler], ids=lambda cls: cls.__name__
)
def test_preempting_scheduled_request_restores_both_budgets(scheduler, monkeypatch):
    requests = [create_request(i, num_tokens=4, block_size=BLOCK_SIZE) for i in range(1, 5)]
    victim, retried, following, deferred = requests
    for request in requests:
        scheduler.add_request(request)
    step(scheduler)
    scheduler.policy = SchedulingPolicy.PRIORITY
    victim.priority = 10
    configure_budgets(scheduler, monkeypatch, token_budget=10, input_budget=6, draft_slots=2)
    allocate_slots = scheduler.kv_cache_manager.allocate_slots
    allocation_failed = False

    def fail_once(request, *args, **kwargs):
        nonlocal allocation_failed
        if request is retried and not allocation_failed:
            allocation_failed = True
            return None
        return allocate_slots(request, *args, **kwargs)

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", fail_once)

    output = scheduler.schedule()

    assert allocation_failed
    assert victim.status == RequestStatus.PREEMPTED
    assert output.num_scheduled_tokens == {retried.request_id: 1, following.request_id: 1}
    assert deferred in scheduler.running


def test_async_kv_load_does_not_consume_drafting_budget(scheduler, monkeypatch):
    loading = create_request(1, num_tokens=3 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    waiting = create_request(2, num_tokens=1, block_size=BLOCK_SIZE)
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        lambda request, num_computed: (2 * BLOCK_SIZE, True) if request is loading else (0, False)
    )
    scheduler.add_request(loading)
    scheduler.add_request(waiting)
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=3, draft_slots=2)

    output = scheduler.schedule()

    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert loading in scheduler.kv_holding_waiting
    assert output.num_scheduled_tokens == {waiting.request_id: 1}


def test_paused_holder_drains_with_input_budget_and_resumes_waiter(scheduler, monkeypatch):
    loading = start_async_load(scheduler, load_done=True)
    waiting = create_request(2, num_tokens=1, max_tokens=2, block_size=BLOCK_SIZE)
    scheduler.add_request(waiting)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    configure_budgets(scheduler, monkeypatch, token_budget=8, input_budget=4, draft_slots=2)

    for iteration in range(MAX_DRAIN_STEPS):
        if not scheduler.has_requests():
            break
        output = scheduler.schedule()
        if iteration == 0:
            assert output.num_scheduled_tokens == {loading.request_id: 2}
        assert waiting.request_id not in output.num_scheduled_tokens
        scheduled = [scheduler.requests[req_id] for req_id in output.num_scheduled_tokens]
        model_output = create_model_runner_output(scheduled)
        for index, request in enumerate(scheduled):
            # A partial prefill produces no sampled token, including with async scheduling.
            if request.is_prefill_chunk:
                model_output.sampled_token_ids[index] = []
        scheduler.update_from_output(output, model_output)

    assert not scheduler.has_requests()
    assert loading.request_id not in scheduler.requests
    assert waiting in scheduler.waiting
    assert not any(scheduler.kv_cache_manager.get_block_ids(waiting.request_id))
    scheduler.set_pause_state(PauseState.UNPAUSED)
    assert step(scheduler).num_scheduled_tokens == {waiting.request_id: 1}
