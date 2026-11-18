# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.core.test_score_encoder_cache_scheduler import _request as make_mm_request
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.utils import vllm_version_is

SCHEDULERS = [
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    RecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    AsyncDyntraLBRecomputeScheduler,
    BalanceScheduler,
]


def _make_scheduler(scheduler_cls):
    config = make_dyntra_test_config(block_size=16)
    config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = False
    scheduler = create_dyntra_lb_scheduler(config, scheduler_cls=scheduler_cls)
    if isinstance(scheduler, BalanceScheduler):
        scheduler._balance_enabled = True
        scheduler.balance_queue = [torch.tensor([0], dtype=torch.int)]
    scheduler.max_num_encoder_input_tokens = 32
    scheduler.encoder_cache_manager = MagicMock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.encoder_cache_manager.get_manager_metadata.return_value = None
    scheduler.ec_connector = None
    scheduler._get_encoder_input_updates = MagicMock(return_value={})

    def schedule_encoder(request, computed, num_tokens, budget, **kwargs):
        inputs = list(range(len(request.mm_features)))
        result = (inputs, num_tokens, budget - sum(request.get_num_encoder_embeds(i) for i in inputs), [])
        # Isolate statistics from the existing four/five-item API difference
        # between copied schedulers; duplicate-cache handling is tested separately.
        return (*result, []) if isinstance(scheduler, RecomputeScheduler) else result

    scheduler._try_schedule_encoder_inputs = schedule_encoder
    return scheduler


def _add_request(scheduler, request, running):
    scheduler.add_request(request)
    if running:
        scheduler.waiting.remove_request(request)
        request.status = RequestStatus.RUNNING
        scheduler.running.append(request)


@pytest.mark.skipif(vllm_version_is("0.33.0"), reason="IterationDetails is a main-only output")
@pytest.mark.parametrize("scheduler_cls", SCHEDULERS)
@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize(
    ("log_stats", "log_details", "profiler", "enabled"),
    [(True, True, None, True), (False, False, "torch", True), (True, False, None, False), (False, True, None, False)],
)
def test_iteration_encoder_counts_and_creation_timing(
    scheduler_cls, running, log_stats, log_details, profiler, enabled
):
    scheduler = _make_scheduler(scheduler_cls)
    scheduler.log_stats = log_stats
    scheduler.observability_config.enable_logging_iteration_details = log_details
    scheduler.vllm_config.profiler_config = SimpleNamespace(profiler=profiler)
    request = make_mm_request("mm", [("first", [True, True]), ("second", [True, True])])
    _add_request(scheduler, request, running)
    implementation = (
        RecomputeScheduler
        if isinstance(scheduler, RecomputeScheduler)
        else (BalanceScheduler if isinstance(scheduler, BalanceScheduler) else DyntraLBScheduler)
    )
    module = sys.modules[implementation.__module__]
    schedule_encoder = scheduler._try_schedule_encoder_inputs
    with patch.object(module, "IterationDetails", wraps=module.IterationDetails) as create_details:

        def check_created_before_encoder(*args, **kwargs):
            assert create_details.call_count == int(enabled)
            return schedule_encoder(*args, **kwargs)

        scheduler._try_schedule_encoder_inputs = check_created_before_encoder
        output = scheduler.schedule()

    assert output.scheduled_encoder_inputs == {"mm": [0, 1]}
    if enabled:
        assert output.iteration_details.num_encoder_inputs == 2
        assert output.iteration_details.num_encoder_output_tokens == 4
    else:
        assert output.iteration_details is None

    scheduler.finish_requests("mm", RequestStatus.FINISHED_ABORTED)
    empty_output = scheduler.schedule()
    if enabled:
        assert empty_output.iteration_details.num_encoder_inputs == 0
        assert empty_output.iteration_details.num_encoder_output_tokens == 0
    else:
        assert empty_output.iteration_details is None


@pytest.mark.skipif(vllm_version_is("0.33.0"), reason="IterationDetails is a main-only output")
@pytest.mark.parametrize("scheduler_cls", SCHEDULERS)
def test_iteration_encoder_counts_remove_preempted_inputs(scheduler_cls):
    scheduler = _make_scheduler(scheduler_cls)
    scheduler.log_stats = True
    scheduler.observability_config.enable_logging_iteration_details = True
    scheduler.policy = SchedulingPolicy.PRIORITY
    first = make_mm_request("first", [("a", [True, True]), ("b", [True, True])])
    second = make_mm_request("second", [("c", [True, True])])
    first.priority = 10
    second.priority = 0
    for request in (first, second):
        _add_request(scheduler, request, running=True)

    allocate = scheduler.kv_cache_manager.allocate_slots
    failed_once = False

    def fail_second_once(request, *args, **kwargs):
        nonlocal failed_once
        if request is second and not failed_once:
            failed_once = True
            return None
        return allocate(request, *args, **kwargs)

    if isinstance(scheduler, RecomputeScheduler):

        def preempt_locally(request, timestamp, **kwargs):
            scheduler._preempt_request(request, timestamp, **kwargs)
            return True

        scheduler._preempt_or_recompute = preempt_locally

    with patch.object(scheduler.kv_cache_manager, "allocate_slots", side_effect=fail_second_once):
        output = scheduler.schedule()

    assert failed_once
    assert first.status == RequestStatus.PREEMPTED
    assert output.scheduled_encoder_inputs == {"second": [0]}
    assert output.iteration_details.num_encoder_inputs == 1
