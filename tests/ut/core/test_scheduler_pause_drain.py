# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.dyntra_lb_scheduler import (
    AsyncDyntraLBScheduler,
    DyntraLBScheduler,
    get_dyntra_lb_request_block_num,
)
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkAsyncScheduler, ProfilingChunkScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

BLOCK_SIZE = 16
NUM_BLOCKS = 64
MAX_DRAIN_STEPS = 16
DYNTRA_SCHEDULERS = (
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    DyntraLBRecomputeScheduler,
    AsyncDyntraLBRecomputeScheduler,
)
SCHEDULERS = (
    *DYNTRA_SCHEDULERS,
    RecomputeScheduler,
    AsyncRecomputeScheduler,
    ProfilingChunkScheduler,
    ProfilingChunkAsyncScheduler,
    BalanceScheduler,
)
SYNC_SCHEDULERS = (RecomputeScheduler, DyntraLBScheduler, ProfilingChunkScheduler, BalanceScheduler)


@pytest.fixture(autouse=True)
def reset_ascend_config() -> Iterator[None]:
    clear_ascend_config()
    yield
    clear_ascend_config()


def make_scheduler(scheduler_cls: type[Scheduler]) -> Scheduler:
    config = make_dyntra_test_config(block_size=BLOCK_SIZE)
    # Use the real cache manager without constructing a transport backend.
    config.kv_transfer_config = None
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    scheduler = create_dyntra_lb_scheduler(config, scheduler_cls, num_blocks=NUM_BLOCKS)
    if isinstance(scheduler, BalanceScheduler):
        # Exercise the copied schedule(), rather than its disabled-path fallback.
        scheduler._balance_enabled = True
        scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]

    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.get_num_new_matched_tokens.return_value = (0, False)
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.request_finished.return_value = (False, None)
    connector.has_pending_push_work.return_value = False
    connector.has_pending_block_frees.return_value = False
    connector.get_kv_connector_stats.return_value = None
    connector.take_events.return_value = ()
    connector.reset_cache.return_value = True
    scheduler.connector = connector
    return scheduler


@pytest.fixture(params=SCHEDULERS, ids=lambda cls: cls.__name__)
def scheduler(request: pytest.FixtureRequest) -> Scheduler:
    return make_scheduler(request.param)


def step(
    scheduler: Scheduler,
    finished_recving: set[str] | None = None,
    *,
    use_eos: bool = False,
) -> SchedulerOutput:
    output = scheduler.schedule()
    scheduled = [scheduler.requests[req_id] for req_id in output.num_scheduled_tokens]
    scheduler.update_from_output(
        output,
        create_model_runner_output(scheduled, finished_recving=finished_recving, use_eos=use_eos),
    )
    return output


def start_async_load(scheduler: Scheduler, *, load_done: bool = False) -> Request:
    loading = create_request(1, num_tokens=3 * BLOCK_SIZE, max_tokens=2, block_size=BLOCK_SIZE)
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        lambda request, num_computed: (2 * BLOCK_SIZE, True) if request is loading else (0, False)
    )
    scheduler.add_request(loading)
    output = step(scheduler, {loading.request_id} if load_done else None)
    assert not output.num_scheduled_tokens
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert loading in scheduler.kv_holding_waiting
    assert any(scheduler.kv_cache_manager.get_block_ids(loading.request_id))
    return loading


def drain(scheduler: Scheduler, queued: Request | None = None) -> None:
    for _ in range(MAX_DRAIN_STEPS):
        if not scheduler.has_requests():
            return
        output = step(scheduler)
        if queued is not None:
            assert queued.request_id not in output.num_scheduled_tokens
            assert not any(scheduler.kv_cache_manager.get_block_ids(queued.request_id))
    pytest.fail("PAUSED_NEW did not drain KV-holding requests")


@pytest.mark.parametrize("load_done_before_pause", [False, True], ids=["load-in-flight", "load-done"])
def test_pause_new_drains_async_kv_loads(scheduler, load_done_before_pause):
    loading = start_async_load(scheduler, load_done=load_done_before_pause)
    queued = create_request(2, num_tokens=BLOCK_SIZE, block_size=BLOCK_SIZE)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.add_request(queued)
    assert scheduler.get_num_unfinished_requests() == 1

    if not load_done_before_pause:
        # The loader is skipped, but the ordinary waiting queue must not be
        # admitted as a fallback after the KV-holding queue is exhausted.
        for finished_recving in (None, {loading.request_id}):
            output = step(scheduler, finished_recving)
            assert not output.num_scheduled_tokens
            assert list(scheduler.kv_holding_waiting) == [loading]
            assert not any(scheduler.kv_cache_manager.get_block_ids(queued.request_id))

    drain(scheduler, queued)

    assert loading.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert loading.request_id not in scheduler.requests
    assert not scheduler.kv_holding_waiting
    assert list(scheduler.waiting) == [queued]
    assert scheduler.get_num_unfinished_requests() == 0
    assert all(block.is_null or block.ref_cnt == 0 for block in scheduler.kv_cache_manager.block_pool.blocks)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    scheduler.connector.get_num_new_matched_tokens.assert_called_once_with(loading, 0)

    scheduler.set_pause_state(PauseState.UNPAUSED)
    assert queued.request_id in step(scheduler).num_scheduled_tokens


def test_pause_new_leaves_blockless_requests_queued(scheduler):
    queued = create_request(1, num_tokens=BLOCK_SIZE, block_size=BLOCK_SIZE)
    scheduler.add_request(queued)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)

    assert scheduler.get_num_unfinished_requests() == 0
    assert not step(scheduler).num_scheduled_tokens
    assert list(scheduler.waiting) == [queued]
    assert not any(scheduler.kv_cache_manager.get_block_ids(queued.request_id))
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", SYNC_SCHEDULERS, ids=lambda cls: cls.__name__)
def test_pause_new_skips_blocked_holder_without_losing_ready_holder(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    blocked = start_async_load(scheduler)
    blocked_blocks = scheduler.kv_cache_manager.get_block_ids(blocked.request_id)
    ready = create_request(2, num_tokens=3 * BLOCK_SIZE, max_tokens=2, block_size=BLOCK_SIZE)
    scheduler.connector.get_num_new_matched_tokens.side_effect = None
    scheduler.connector.get_num_new_matched_tokens.return_value = (2 * BLOCK_SIZE, True)
    scheduler.add_request(ready)
    step(scheduler, {ready.request_id})
    scheduler.set_pause_state(PauseState.PAUSED_NEW)

    output = step(scheduler)

    assert set(output.num_scheduled_tokens) == {ready.request_id}
    assert list(scheduler.kv_holding_waiting) == [blocked]
    assert blocked.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.kv_cache_manager.get_block_ids(blocked.request_id) == blocked_blocks

    step(scheduler, {blocked.request_id})
    drain(scheduler)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)


@pytest.mark.parametrize("scheduler_cls", SYNC_SCHEDULERS, ids=lambda cls: cls.__name__)
def test_unpaused_drains_holders_before_admitting_blockless_requests(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    loading = start_async_load(scheduler, load_done=True)
    queued = create_request(2, num_tokens=BLOCK_SIZE, block_size=BLOCK_SIZE)
    scheduler.add_request(queued)

    assert list(step(scheduler).num_scheduled_tokens) == [loading.request_id, queued.request_id]


def test_pause_all_keeps_loaded_request_frozen(scheduler):
    loading = start_async_load(scheduler, load_done=True)
    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert not step(scheduler).num_scheduled_tokens
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert list(scheduler.kv_holding_waiting) == [loading]

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    drain(scheduler)
    assert loading.request_id not in scheduler.requests


def test_pause_new_ignores_streaming_session_waiting_for_input(scheduler):
    session = create_request(1, num_tokens=BLOCK_SIZE, max_tokens=4, block_size=BLOCK_SIZE)
    session.resumable = True
    scheduler.add_request(session)
    step(scheduler, use_eos=True)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert session in scheduler.kv_holding_waiting

    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    assert scheduler.get_num_unfinished_requests() == 0
    assert not step(scheduler).num_scheduled_tokens


@pytest.mark.parametrize("scheduler_cls", DYNTRA_SCHEDULERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("pause_state", list(PauseState))
def test_dyntra_prefetch_obeys_pause(scheduler_cls, pause_state):
    scheduler = make_scheduler(scheduler_cls)
    queued = create_request(1, num_tokens=3 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    scheduler.add_request(queued)
    scheduler.connector.get_num_new_matched_tokens.return_value = (2 * BLOCK_SIZE, True)
    scheduler._lb_kv_prefetch_enabled = True
    scheduler.set_pause_state(pause_state)

    scheduler.prepare_dyntra_lb_step()

    if pause_state == PauseState.UNPAUSED:
        assert queued.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert queued in scheduler.kv_holding_waiting
        assert any(scheduler.kv_cache_manager.get_block_ids(queued.request_id))
        scheduler.connector.update_state_after_alloc.assert_called_once()
    else:
        assert queued.status == RequestStatus.WAITING
        assert list(scheduler.waiting) == [queued]
        assert not any(scheduler.kv_cache_manager.get_block_ids(queued.request_id))
        scheduler.connector.get_num_new_matched_tokens.assert_not_called()
        scheduler.connector.update_state_after_alloc.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", DYNTRA_SCHEDULERS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("lb_paused", [False, True], ids=["remote-load", "lb-paused"])
def test_dyntra_pause_drains_kv_holders_despite_frozen_plan(scheduler_cls, lb_paused):
    scheduler = make_scheduler(scheduler_cls)
    if lb_paused:
        holder = create_request(1, num_tokens=BLOCK_SIZE, max_tokens=2, block_size=BLOCK_SIZE)
        scheduler.add_request(holder)
        step(scheduler)
        scheduler.running.remove(holder)
        scheduler._lb_pause_request(holder, 0.0)
    else:
        holder = start_async_load(scheduler, load_done=True)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.modifications = {"out_blk": [], "in_blk": [], "freeze": True}

    drain(scheduler)

    assert holder.request_id not in scheduler.requests
    assert scheduler.modifications is None
    assert scheduler._lb_admit_req_ids is None
    assert not scheduler.lb_freeze
    assert scheduler.reset_prefix_cache(reset_running_requests=True)


@pytest.mark.parametrize("scheduler_cls", DYNTRA_SCHEDULERS, ids=lambda cls: cls.__name__)
def test_dyntra_pause_all_does_not_apply_outgoing_plan(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    running = create_request(1, num_tokens=BLOCK_SIZE, block_size=BLOCK_SIZE)
    scheduler.add_request(running)
    step(scheduler)
    blocks = scheduler.kv_cache_manager.get_block_ids(running.request_id)
    scheduler.modifications = {
        "out_blk": [get_dyntra_lb_request_block_num(scheduler, running)],
        "in_blk": [],
        "freeze": False,
    }
    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert not step(scheduler).num_scheduled_tokens
    assert scheduler.running == [running]
    assert running.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.get_block_ids(running.request_id) == blocks
    assert scheduler.modifications is None


def test_balance_pause_still_obeys_rank_capacity():
    # PAUSED_NEW intentionally retains the global capacity freeze for KV holders.
    scheduler = make_scheduler(BalanceScheduler)
    loading = start_async_load(scheduler, load_done=True)
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    scheduler.balance_queue = [torch.tensor([scheduler.max_num_running_reqs], dtype=torch.int)]

    assert not step(scheduler).num_scheduled_tokens
    assert loading in scheduler.kv_holding_waiting

    # Simulate balance_gather() observing freed capacity during a coordinated pause.
    scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]
    drain(scheduler)
    assert loading.request_id not in scheduler.requests
