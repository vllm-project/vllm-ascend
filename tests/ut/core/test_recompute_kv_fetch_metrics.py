# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV-fetch metrics lifecycle tests; no model or worker execution."""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler


@pytest.fixture(params=[RecomputeScheduler, AsyncRecomputeScheduler])
def scheduler_factory(request):
    def build(stats=True):
        config = make_dyntra_test_config(max_num_seqs=2)
        original = Scheduler.__init__

        def initialize(self, *args, **kwargs):
            kwargs["log_stats"] = stats
            original(self, *args, **kwargs)

        with patch.object(Scheduler, "__init__", initialize):
            scheduler = create_dyntra_lb_scheduler(config, scheduler_cls=request.param)
        req = create_request(request_id=71, do_remote_prefill=True, block_size=config.cache_config.block_size)
        scheduler.add_request(req)
        return scheduler, req

    return build


def assert_counts(s, waiting=0, active=0, done=0):
    assert s._kv_fetch_counts == dict(waiting_to_start=waiting, in_progress=active, completed_waiting=done)


def complete_receive(s, r):
    s._update_from_kv_xfer_finished(SimpleNamespace(finished_recving={r.request_id}, finished_sending=set()))


def start_loading(s, r):
    s.schedule()
    assert r.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("phase", ["queued", "loading", "completed"])
def test_cancel_clears_metrics_even_with_late_receive(scheduler_factory, phase):
    s, r = scheduler_factory()
    assert_counts(s, waiting=1)
    if phase != "queued":
        start_loading(s, r)
        assert_counts(s, active=1)
    if phase == "completed":
        complete_receive(s, r)
        assert_counts(s, done=1)
    s.finish_requests(r.request_id, RequestStatus.FINISHED_ABORTED)
    assert_counts(s)
    if phase == "loading":
        assert r.request_id in s.requests
        complete_receive(s, r)
        assert_counts(s)
        assert r.request_id not in s.requests


def test_duplicate_completion_does_not_double_count(scheduler_factory):
    s, r = scheduler_factory()
    start_loading(s, r)
    complete_receive(s, r)
    complete_receive(s, r)
    assert_counts(s, done=1)
    s.schedule()
    assert r.status == RequestStatus.RUNNING
    assert_counts(s)


@pytest.mark.parametrize("policy", ["fail", "recompute"])
def test_failed_receive_cleanup(scheduler_factory, policy):
    s, r = scheduler_factory()
    start_loading(s, r)
    # Inject the state produced by failed-load detection, not an actual transport.
    s.failed_recving_kv_req_ids.add(r.request_id)
    r.num_computed_tokens = 0
    complete_receive(s, r)
    assert_counts(s, done=1)
    if policy == "fail":
        s.finish_requests(r.request_id, RequestStatus.FINISHED_ERROR)
    else:
        with patch.object(s.connector, "get_num_new_matched_tokens", return_value=(0, False)):
            s.schedule()
        assert r.status == RequestStatus.RUNNING
    assert_counts(s)


def test_offloaded_preemption_reload(scheduler_factory):
    s, r = scheduler_factory()
    start_loading(s, r)
    complete_receive(s, r)
    scheduled = s.schedule()
    assert r.status == RequestStatus.RUNNING
    assert_counts(s)
    # Complete the scheduled model step before preemption; otherwise the
    # scheduler correctly waits for deliverable stale output.
    s.update_from_output(scheduled, create_model_runner_output([r]))
    assert r.num_in_flight_tokens == 0
    s.running.remove(r)
    with patch.object(s.connector, "update_state_before_preempt", return_value=True, create=True):
        assert s._preempt_or_recompute(r, time.monotonic())
    assert r.status == RequestStatus.PREEMPTED
    assert_counts(s)
    # Model a connector cache hit on the next allocation after CPU offload.
    with patch.object(s.connector, "get_num_new_matched_tokens", return_value=(r.num_tokens - 1, True)):
        s.schedule()
    assert r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert_counts(s, active=1)
    complete_receive(s, r)
    assert_counts(s, done=1)
    s.schedule()
    assert r.status == RequestStatus.RUNNING
    assert_counts(s)


def test_metrics_disabled(scheduler_factory):
    s, r = scheduler_factory(stats=False)
    assert s._kv_fetch_stages is None
    start_loading(s, r)
    complete_receive(s, r)
    s.schedule()
    assert r.status == RequestStatus.RUNNING
    s.finish_requests(r.request_id, RequestStatus.FINISHED_ABORTED)
    assert_counts(s)


def test_pause_retains_real_wait_until_resume(scheduler_factory):
    s, r = scheduler_factory()
    start_loading(s, r)
    s.set_pause_state(PauseState.PAUSED_ALL)
    complete_receive(s, r)
    s.schedule()
    assert r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert_counts(s, done=1)
    s.set_pause_state(PauseState.UNPAUSED)
    s.schedule()
    assert r.status == RequestStatus.RUNNING
    assert_counts(s)


def test_loading_stage(scheduler_factory):
    scheduler, req = scheduler_factory()
    assert_counts(scheduler, waiting=1)
    start_loading(scheduler, req)
    assert_counts(scheduler, active=1)


def test_running_clears_completed_wait(scheduler_factory):
    scheduler, req = scheduler_factory()
    start_loading(scheduler, req)
    complete_receive(scheduler, req)
    assert_counts(scheduler, done=1)
    scheduler.schedule()
    assert req.status == RequestStatus.RUNNING
    assert_counts(scheduler)
